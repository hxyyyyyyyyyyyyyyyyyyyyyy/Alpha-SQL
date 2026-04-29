#!/usr/bin/env python3
import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


QUESTION_RE = re.compile(r"Question ID:\s*(\d+)\s+done(?:,\s*valid reasoning paths:\s*(\d+))?", re.IGNORECASE)
PROMPT_RE = re.compile(r"Total prompt tokens:\s*(\d+)", re.IGNORECASE)
COMPLETION_RE = re.compile(r"Total completion tokens:\s*(\d+)", re.IGNORECASE)
TOTAL_RE = re.compile(r"Total tokens:\s*(\d+)", re.IGNORECASE)
COST_RE = re.compile(r"Total cost:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


@dataclass
class Record:
    index: int
    line_no: int
    question_id: int
    valid_reasoning_paths: Optional[int]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    total_cost: Optional[float]


def _record_total_tokens(record: Record) -> Optional[int]:
    if record.total_tokens is not None:
        return record.total_tokens
    if record.prompt_tokens is not None and record.completion_tokens is not None:
        return record.prompt_tokens + record.completion_tokens
    return None


def _assign_worker_and_delta(records: List[Record]) -> List[Dict]:
    """
    Estimate per-question token usage from cumulative logs by assigning each record
    to a monotonic worker stream and taking deltas inside that stream.
    """
    workers: List[Dict[str, Optional[float]]] = []
    output: List[Dict] = []

    for rec in records:
        rec_total = _record_total_tokens(rec)

        best_idx = -1
        best_score = None

        if rec_total is not None:
            for idx, worker in enumerate(workers):
                worker_total = worker.get("last_total")
                if worker_total is None:
                    continue
                if rec_total >= worker_total:
                    delta = rec_total - worker_total
                    if best_score is None or delta < best_score:
                        best_score = delta
                        best_idx = idx

        if best_idx < 0:
            workers.append(
                {
                    "last_prompt": rec.prompt_tokens,
                    "last_completion": rec.completion_tokens,
                    "last_total": rec_total,
                    "last_cost": rec.total_cost,
                }
            )
            worker_id = len(workers) - 1
            worker_start = True
            delta_prompt = rec.prompt_tokens
            delta_completion = rec.completion_tokens
            delta_total = rec_total
            delta_cost = rec.total_cost
        else:
            worker = workers[best_idx]
            worker_id = best_idx
            worker_start = False

            last_prompt = worker.get("last_prompt")
            last_completion = worker.get("last_completion")
            last_total = worker.get("last_total")
            last_cost = worker.get("last_cost")

            delta_prompt = (
                rec.prompt_tokens - int(last_prompt)
                if rec.prompt_tokens is not None and last_prompt is not None and rec.prompt_tokens >= int(last_prompt)
                else rec.prompt_tokens
            )
            delta_completion = (
                rec.completion_tokens - int(last_completion)
                if rec.completion_tokens is not None
                and last_completion is not None
                and rec.completion_tokens >= int(last_completion)
                else rec.completion_tokens
            )
            delta_total = (
                rec_total - int(last_total)
                if rec_total is not None and last_total is not None and rec_total >= int(last_total)
                else rec_total
            )
            delta_cost = (
                rec.total_cost - float(last_cost)
                if rec.total_cost is not None and last_cost is not None and rec.total_cost >= float(last_cost)
                else rec.total_cost
            )

            worker["last_prompt"] = rec.prompt_tokens
            worker["last_completion"] = rec.completion_tokens
            worker["last_total"] = rec_total
            worker["last_cost"] = rec.total_cost

        output.append(
            {
                "index": rec.index,
                "line_no": rec.line_no,
                "question_id": rec.question_id,
                "valid_reasoning_paths": rec.valid_reasoning_paths,
                "prompt_tokens": rec.prompt_tokens,
                "completion_tokens": rec.completion_tokens,
                "total_tokens": rec.total_tokens,
                "total_cost": rec.total_cost,
                "estimated_delta_prompt_tokens": delta_prompt,
                "estimated_delta_completion_tokens": delta_completion,
                "estimated_delta_total_tokens": delta_total,
                "estimated_delta_cost": delta_cost,
                "worker_id": worker_id,
                "worker_start": worker_start,
            }
        )

    return output


def _parse_log_file(file_path: Path) -> Tuple[List[Record], Dict[str, int]]:
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    records: List[Record] = []
    error_counts = {
        "schema_selection_parse_errors": 0,
        "sql_generation_parse_errors": 0,
        "instr_occurrence_errors": 0,
        "named_columns_alias_errors": 0,
    }

    for line in lines:
        if "Error parsing schema selection response" in line:
            error_counts["schema_selection_parse_errors"] += 1
        if "Error parsing sql generation response" in line:
            error_counts["sql_generation_parse_errors"] += 1
        if "INSTR does not support the occurrence parameter" in line:
            error_counts["instr_occurrence_errors"] += 1
        if "Named columns are not supported in table alias" in line:
            error_counts["named_columns_alias_errors"] += 1

    i = 0
    while i < len(lines):
        line = lines[i]

        q_match = QUESTION_RE.search(line)
        if not q_match:
            i += 1
            continue

        qid = int(q_match.group(1))
        vrp = int(q_match.group(2)) if q_match.group(2) is not None else None

        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        total_cost = None

        j = i + 1
        while j < len(lines):
            next_line = lines[j]
            if QUESTION_RE.search(next_line):
                break

            p = PROMPT_RE.search(next_line)
            if p:
                prompt_tokens = int(p.group(1))

            c = COMPLETION_RE.search(next_line)
            if c:
                completion_tokens = int(c.group(1))

            t = TOTAL_RE.search(next_line)
            if t:
                total_tokens = int(t.group(1))

            co = COST_RE.search(next_line)
            if co:
                total_cost = float(co.group(1))

            # Most blocks end quickly; cap lookahead to avoid scanning too far
            if j - i > 25:
                break
            j += 1

        records.append(
            Record(
                index=len(records),
                line_no=i + 1,
                question_id=qid,
                valid_reasoning_paths=vrp,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                total_cost=total_cost,
            )
        )

        i = j

    return records, error_counts


def process_file(file_path: Path) -> Path:
    records, error_counts = _parse_log_file(file_path)
    node_items = _assign_worker_and_delta(records)

    estimated_prompt_sum = sum(v["estimated_delta_prompt_tokens"] or 0 for v in node_items)
    estimated_completion_sum = sum(v["estimated_delta_completion_tokens"] or 0 for v in node_items)
    estimated_total_sum = sum(v["estimated_delta_total_tokens"] or 0 for v in node_items)
    estimated_cost_sum = sum(v["estimated_delta_cost"] or 0.0 for v in node_items)

    output = {
        "source_file": str(file_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "question_node_count": len(node_items),
            "workers_estimated": len(set(v["worker_id"] for v in node_items)) if node_items else 0,
            "estimated_total_prompt_tokens": estimated_prompt_sum,
            "estimated_total_completion_tokens": estimated_completion_sum,
            "estimated_total_tokens": estimated_total_sum,
            "estimated_total_cost": estimated_cost_sum,
        },
        "error_counts": error_counts,
        "node_token_usage": node_items,
    }

    out_path = file_path.with_suffix(".json")
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def collect_targets(single_file: Optional[str], log_dir: str) -> List[Path]:
    if single_file:
        path = Path(single_file)
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {single_file}")
        if path.is_dir():
            raise IsADirectoryError(f"Expected a file but got directory: {single_file}")
        return [path.resolve()]

    base = Path(log_dir)
    if not base.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    targets = [p.resolve() for p in base.rglob("*.log") if p.is_file()]
    return sorted(targets)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计日志中每个 Question 节点的 token 消耗，并输出同名 JSON 文件。"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="指定单个日志文件路径；不指定时批量处理 --log-dir 下全部 .log 文件",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="log",
        help="批量处理时的日志目录（默认: log）",
    )

    args = parser.parse_args()
    targets = collect_targets(args.file, args.log_dir)

    if not targets:
        print("No .log files found to process.")
        return

    print(f"Found {len(targets)} log file(s) to process.")
    for path in targets:
        out = process_file(path)
        print(f"Generated: {out}")


if __name__ == "__main__":
    main()
