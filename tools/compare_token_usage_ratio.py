#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _to_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_delta_or_raw(item: Dict[str, Any], delta_key: str, raw_key: str) -> Optional[float]:
    delta_val = _to_number(item.get(delta_key))
    if delta_val is not None:
        return delta_val
    return _to_number(item.get(raw_key))


def _aggregate_by_question(json_path: Path) -> Dict[int, Dict[str, float]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = payload.get("node_token_usage", [])

    agg: Dict[int, Dict[str, float]] = {}
    for row in rows:
        qid = row.get("question_id")
        if qid is None:
            continue

        try:
            qid_int = int(qid)
        except (TypeError, ValueError):
            continue

        prompt = _pick_delta_or_raw(row, "estimated_delta_prompt_tokens", "prompt_tokens")
        response = _pick_delta_or_raw(row, "estimated_delta_completion_tokens", "completion_tokens")
        total = _pick_delta_or_raw(row, "estimated_delta_total_tokens", "total_tokens")

        if qid_int not in agg:
            agg[qid_int] = {
                "prompt_tokens": 0.0,
                "response_tokens": 0.0,
                "total_tokens": 0.0,
                "row_count": 0.0,
                "valid_prompt": 0.0,
                "valid_response": 0.0,
                "valid_total": 0.0,
            }

        item = agg[qid_int]
        item["row_count"] += 1.0

        if prompt is not None:
            item["prompt_tokens"] += prompt
            item["valid_prompt"] += 1.0
        if response is not None:
            item["response_tokens"] += response
            item["valid_response"] += 1.0
        if total is not None:
            item["total_tokens"] += total
            item["valid_total"] += 1.0

    return agg


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def compare(
    file_a: Path,
    file_b: Path,
    name_a: str,
    name_b: str,
    output_json: Path,
    output_csv: Path,
) -> Dict[str, Any]:
    agg_a = _aggregate_by_question(file_a)
    agg_b = _aggregate_by_question(file_b)

    all_qids = sorted(set(agg_a.keys()) | set(agg_b.keys()))

    per_point: List[Dict[str, Any]] = []

    matched_prompt_a = 0.0
    matched_response_a = 0.0
    matched_total_a = 0.0

    matched_prompt_b = 0.0
    matched_response_b = 0.0
    matched_total_b = 0.0

    matched_count = 0
    ignored_one_side_count = 0

    for qid in all_qids:
        a = agg_a.get(qid)
        b = agg_b.get(qid)

        a_prompt = a["prompt_tokens"] if a and a["valid_prompt"] > 0 else None
        a_response = a["response_tokens"] if a and a["valid_response"] > 0 else None
        a_total = a["total_tokens"] if a and a["valid_total"] > 0 else None

        b_prompt = b["prompt_tokens"] if b and b["valid_prompt"] > 0 else None
        b_response = b["response_tokens"] if b and b["valid_response"] > 0 else None
        b_total = b["total_tokens"] if b and b["valid_total"] > 0 else None

        both_have_total = a_total is not None and b_total is not None
        if both_have_total:
            matched_count += 1
            matched_prompt_a += a_prompt or 0.0
            matched_response_a += a_response or 0.0
            matched_total_a += a_total

            matched_prompt_b += b_prompt or 0.0
            matched_response_b += b_response or 0.0
            matched_total_b += b_total
        else:
            if (a_total is None) != (b_total is None):
                ignored_one_side_count += 1

        per_point.append(
            {
                "question_id": qid,
                f"{name_a}_prompt_tokens": a_prompt,
                f"{name_a}_response_tokens": a_response,
                f"{name_a}_total_tokens": a_total,
                f"{name_b}_prompt_tokens": b_prompt,
                f"{name_b}_response_tokens": b_response,
                f"{name_b}_total_tokens": b_total,
                "used_in_total_ratio": both_have_total,
            }
        )

    ratio_total_a_over_b = _safe_ratio(matched_total_a, matched_total_b)
    ratio_total_b_over_a = _safe_ratio(matched_total_b, matched_total_a)

    summary = {
        "method_a": name_a,
        "method_b": name_b,
        "file_a": str(file_a),
        "file_b": str(file_b),
        "question_count_a": len(agg_a),
        "question_count_b": len(agg_b),
        "question_count_union": len(all_qids),
        "question_count_matched_for_total": matched_count,
        "ignored_one_side_points_for_total": ignored_one_side_count,
        "matched_totals": {
            name_a: {
                "prompt_tokens": matched_prompt_a,
                "response_tokens": matched_response_a,
                "total_tokens": matched_total_a,
            },
            name_b: {
                "prompt_tokens": matched_prompt_b,
                "response_tokens": matched_response_b,
                "total_tokens": matched_total_b,
            },
        },
        "ratios": {
            f"{name_a}_over_{name_b}": ratio_total_a_over_b,
            f"{name_b}_over_{name_a}": ratio_total_b_over_a,
        },
    }

    output = {
        "summary": summary,
        "per_question": per_point,
    }

    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "question_id",
        f"{name_a}_prompt_tokens",
        f"{name_a}_response_tokens",
        f"{name_a}_total_tokens",
        f"{name_b}_prompt_tokens",
        f"{name_b}_response_tokens",
        f"{name_b}_total_tokens",
        "used_in_total_ratio",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_point:
            writer.writerow(row)

    print(f"Output JSON: {output_json}")
    print(f"Output CSV: {output_csv}")
    print(f"Matched points used in ratio: {matched_count}")
    print(f"Ignored one-side points: {ignored_one_side_count}")
    print(
        f"Total ratio ({name_a}/{name_b}) = {_format_ratio(ratio_total_a_over_b)}, "
        f"({name_b}/{name_a}) = {_format_ratio(ratio_total_b_over_a)}"
    )

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="比较两个 token 统计 JSON，输出总消耗比值和每个数据点的 token 对比。"
    )
    parser.add_argument("--file-a", default="log/llm_nodescore_bird.json", help="方法A JSON路径")
    parser.add_argument("--file-b", default="log/origin_bird_dev.json", help="方法B JSON路径")
    parser.add_argument("--name-a", default="llm_nodescore", help="方法A名称")
    parser.add_argument("--name-b", default="origin", help="方法B名称")
    parser.add_argument(
        "--output-json",
        default="log/token_usage_ratio_llm_nodescore_vs_origin.json",
        help="输出JSON路径",
    )
    parser.add_argument(
        "--output-csv",
        default="log/token_usage_ratio_llm_nodescore_vs_origin.csv",
        help="输出CSV路径",
    )
    args = parser.parse_args()

    file_a = Path(args.file_a).resolve()
    file_b = Path(args.file_b).resolve()
    output_json = Path(args.output_json).resolve()
    output_csv = Path(args.output_csv).resolve()

    if not file_a.exists():
        raise FileNotFoundError(f"file_a not found: {file_a}")
    if not file_b.exists():
        raise FileNotFoundError(f"file_b not found: {file_b}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    compare(
        file_a=file_a,
        file_b=file_b,
        name_a=args.name_a,
        name_b=args.name_b,
        output_json=output_json,
        output_csv=output_csv,
    )


if __name__ == "__main__":
    main()
