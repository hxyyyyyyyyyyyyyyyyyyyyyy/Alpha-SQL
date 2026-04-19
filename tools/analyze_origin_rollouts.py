#!/usr/bin/env python3
"""Analyze origin rollout PKL files and export per-rollout SQL correctness JSON.

The SQL correctness criterion is intentionally aligned with
`alphasql/runner/evaluation.py`:
- Execute predicted SQL and golden SQL on the same sqlite database.
- Compare `set(predicted_res) == set(ground_truth_res)`.
- Timeout and execution errors are judged as incorrect.
"""

import argparse
import importlib
import json
import pickle
import re
import sqlite3
import sys
import types
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

from func_timeout import FunctionTimedOut, func_timeout


def _ensure_project_root_on_sys_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def _ensure_unpickle_stubs() -> None:
    """Inject lightweight stubs for optional runtime-only dependencies.

    Some pickle payloads reference classes that import online LLM clients.
    We only need class loading for unpickling, not real API calls.
    """
    try:
        importlib.import_module("openai")
    except Exception:
        openai_module = types.ModuleType("openai")

        class OpenAI:  # pragma: no cover - runtime stub
            pass

        openai_module.OpenAI = OpenAI
        sys.modules["openai"] = openai_module

    try:
        importlib.import_module("dotenv")
    except Exception:
        dotenv_module = types.ModuleType("dotenv")
        dotenv_module.load_dotenv = lambda override=True: None
        sys.modules["dotenv"] = dotenv_module


def load_ground_truth_map(ground_truth_path: Path) -> Dict[int, Tuple[str, str]]:
    """Load `{question_id: (gold_sql, db_name)}` from Bird-style `.sql` file."""
    mapping: Dict[int, Tuple[str, str]] = {}
    with ground_truth_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            mapping[i] = (parts[0], parts[1])
    return mapping


def execute_sql_same_standard(predicted_sql: str, ground_truth: str, db_path: Path) -> int:
    """Same correctness criterion as `alphasql/runner/evaluation.py`.

    Return 1 when results are equal by set comparison, otherwise 0.
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    try:
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        return 1 if set(predicted_res) == set(ground_truth_res) else 0
    finally:
        conn.close()


def judge_sql(
    predicted_sql: str,
    ground_truth_sql: str,
    db_path: Path,
    timeout_s: float,
) -> Tuple[int, str]:
    """Evaluate SQL and return `(result, error_msg)`.

    `result`: 1 correct, 0 incorrect
    `error_msg`: "" when executed normally (correct or incorrect result-set),
    otherwise timeout / error message.
    """
    if not predicted_sql or not predicted_sql.strip():
        return 0, "empty_sql"

    try:
        res = func_timeout(
            timeout_s,
            execute_sql_same_standard,
            args=(predicted_sql, ground_truth_sql, db_path),
        )
        return int(res), ""
    except FunctionTimedOut:
        return 0, "timeout"
    except Exception as e:  # noqa: BLE001
        return 0, f"error: {type(e).__name__}: {e}"


def extract_actions(rollout_nodes: Sequence[Any]) -> List[str]:
    actions: List[str] = []
    for node in rollout_nodes:
        parent_action = getattr(node, "parent_action", None)
        if parent_action is None:
            continue
        actions.append(parent_action.__class__.__name__)
    return actions


def _strip_think_content(text: str) -> str:
    if not text:
        return ""
    # Remove both <think>...</think> and <think>...<\think> blocks.
    cleaned = re.sub(r"<think\b[^>]*>.*?(?:</think>|<\\think>)", "", text, flags=re.DOTALL | re.IGNORECASE)
    # If there is an unclosed <think>, drop content from it to the end.
    cleaned = re.sub(r"<think\b[^>]*>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</think>|<\\think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_action_response(node: Any, action_name: str) -> str:
    """Extract response text for a node by action type.

    Some actions do not persist the original raw LLM response in the node,
    so we fall back to the closest stored output.
    """
    
    if action_name == "RaphraseQuestionAction":
        return _strip_think_content(getattr(node, "rephrased_question", "") or "")
    if action_name == "SchemaSelectionAction":
        return _strip_think_content(getattr(node, "selected_schema_context", "") or "")
    if action_name == "IdentifyColumnValuesAction":
        return _strip_think_content(getattr(node, "identified_column_values", "") or "")
    if action_name == "IdentifyColumnFunctionsAction":
        return _strip_think_content(getattr(node, "identified_column_functions", "") or "")
    if action_name == "SQLGenerationAction":
        return _strip_think_content(getattr(node, "sql_query", "") or "")
    if action_name == "SQLRevisionAction":
        return _strip_think_content(getattr(node, "revised_sql_query", "") or "")
    return ""


def extract_action_details(rollout_nodes: Sequence[Any]) -> List[Dict[str, str]]:
    details: List[Dict[str, str]] = []
    for node in rollout_nodes:
        parent_action = getattr(node, "parent_action", None)
        if parent_action is None:
            continue
        action_name = parent_action.__class__.__name__
        detail_item: Dict[str, str] = {
            "action": action_name,
            "response": _extract_action_response(node, action_name),
        }
        details.append(detail_item)
    return details


def extract_final_sql(rollout_nodes: Sequence[Any]) -> str:
    """Get final SQL from rollout by checking end-to-start with priority.

    Priority: `final_sql_query` > `revised_sql_query` > `sql_query`.
    """
    for node in reversed(rollout_nodes):
        for key in ("final_sql_query", "revised_sql_query", "sql_query"):
            value = getattr(node, key, None)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def find_pkl_files(results_dir: Path) -> List[Path]:
    return sorted(results_dir.rglob("*.pkl"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def infer_dataset_number(pkl_path: Path, root_results_dir: Path) -> str:
    """Infer dataset_number for key `pkl_number(dataset_number)`.

    We use the pkl index when it is numeric, keeping compatibility with Bird dev
    where pkl filename equals question index.
    """
    if pkl_path.stem.isdigit():
        return pkl_path.stem
    return pkl_path.relative_to(root_results_dir).as_posix()


def build_sample_key(pkl_path: Path, root_results_dir: Path) -> str:
    """Build output key and avoid duplicated pkl number like `0(0)`.

    If inferred dataset number equals pkl stem, only keep pkl stem once.
    """
    pkl_number = pkl_path.stem
    dataset_number = infer_dataset_number(pkl_path, root_results_dir)
    if dataset_number == pkl_number:
        return pkl_number
    return f"{pkl_number}({dataset_number})"


def add_detail_suffix(path: Path, detail: bool) -> Path:
    if not detail:
        return path
    if path.name.endswith("_detail"):
        return path
    return path.with_name(f"{path.name}_detail")


def load_completed_qids(output_dir: Path) -> set[int]:
    """Read existing per-point JSON files and collect completed qids.

    A file is treated as completed when it is a valid JSON object that contains
    the expected qid key written by this script.
    """
    completed_qids: set[int] = set()
    if not output_dir.exists():
        return completed_qids

    for json_path in output_dir.glob("*.json"):
        if not json_path.stem.isdigit():
            continue

        qid = int(json_path.stem)
        expected_key = str(qid)

        try:
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:  # noqa: BLE001
            continue

        if isinstance(payload, dict) and expected_key in payload:
            completed_qids.add(qid)

    return completed_qids


def analyze_rollouts(
    results_dir: Path,
    ground_truth_map: Dict[int, Tuple[str, str]],
    db_root_path: Path,
    timeout_s: float,
    num_workers: int,
    output_dir: Path,
    detail: bool,
    max_pkl: Optional[int] = None,
) -> int:
    _ensure_project_root_on_sys_path()
    _ensure_unpickle_stubs()
    output_dir = add_detail_suffix(output_dir, detail)

    pkl_files = find_pkl_files(results_dir)
    valid_pkl_files: List[Path] = []
    for pkl_path in pkl_files:
        if not pkl_path.stem.isdigit():
            continue
        if int(pkl_path.stem) not in ground_truth_map:
            continue
        valid_pkl_files.append(pkl_path)

    completed_qids = load_completed_qids(output_dir)
    if completed_qids:
        valid_pkl_files = [p for p in valid_pkl_files if int(p.stem) not in completed_qids]

    if max_pkl is not None:
        valid_pkl_files = valid_pkl_files[:max_pkl]

    total_pkl = len(valid_pkl_files)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Start processing {total_pkl} pkl files with {num_workers} threads"
        + (f" (skipped {len(completed_qids)} completed files)" if completed_qids else ""),
        flush=True,
    )

    def _process_one_pkl(pkl_path: Path) -> Tuple[int, str, Path]:
        qid = int(pkl_path.stem)
        golden_sql, db_name = ground_truth_map[qid]
        db_path = db_root_path / db_name / f"{db_name}.sqlite"
        key = build_sample_key(pkl_path, results_dir)

        try:
            with pkl_path.open("rb") as f:
                rollout_list = pickle.load(f)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            point_obj: Dict[str, Any] = {
                "golden_sql": golden_sql,
                "load_error": err,
            }
            point_file = output_dir / f"{qid}.json"
            with point_file.open("w", encoding="utf-8") as f:
                json.dump({key: point_obj}, f, ensure_ascii=False, indent=4)
            return qid, "load_error", point_file

        item: Dict[str, Any] = {"golden_sql": golden_sql}

        if not isinstance(rollout_list, list):
            item["rollout_0"] = {
                "actions": [],
                "SQL": "",
                "result": 0,
                "error_msg": "invalid_rollout_format",
            }
            point_file = output_dir / f"{qid}.json"
            with point_file.open("w", encoding="utf-8") as f:
                json.dump({key: item}, f, ensure_ascii=False, indent=4)
            return qid, "invalid_rollout_format", point_file

        for rollout_idx, rollout_nodes in enumerate(rollout_list):
            if not isinstance(rollout_nodes, list):
                item[str(rollout_idx)] = {
                    "actions": [],
                    "SQL": "",
                    "result": 0,
                    "error_msg": "invalid_rollout_nodes_format",
                }
                continue

            actions = extract_actions(rollout_nodes)
            sql = extract_final_sql(rollout_nodes)
            result, error_msg = judge_sql(sql, golden_sql, db_path, timeout_s)

            item[str(rollout_idx)] = {
                "actions": actions,
                "SQL": sql,
                "result": result,
                "error_msg": error_msg,
            }
            if detail:
                item[str(rollout_idx)]["action_responses"] = extract_action_details(rollout_nodes)

        point_file = output_dir / f"{qid}.json"
        with point_file.open("w", encoding="utf-8") as f:
            json.dump({key: item}, f, ensure_ascii=False, indent=4)
        return qid, "ok", point_file

    done_count = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_one_pkl, p) for p in valid_pkl_files]
        for future in as_completed(futures):
            qid, status, point_file = future.result()
            done_count += 1
            print(
                f"[PKL DONE] qid={qid}, status={status}, result_file={point_file}, progress={done_count}/{total_pkl}",
                flush=True,
            )

    return done_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze origin rollout PKL files")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/origin/Qwen2.5-Coder-7B-Instruct/bird/dev"),
        help="Directory containing rollout pkl files",
    )
    parser.add_argument(
        "--ground_truth_path",
        type=Path,
        default=Path("data/bird/dev/dev.sql"),
        help="Ground-truth SQL file",
    )
    parser.add_argument(
        "--db_root_path",
        type=Path,
        default=Path("data/bird/dev/dev_databases"),
        help="Database root path (contains db_name/db_name.sqlite)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-SQL execution timeout in seconds",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/origin/Qwen2.5-Coder-7B-Instruct/rollout_analysis_points"),
        help="Directory for per-data-point result json files",
    )
    parser.add_argument(
        "--max_pkl",
        type=int,
        default=None,
        help="Only process first N pkl files (for quick debug)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker threads for concurrent pkl processing",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Include per-action response details in each rollout output",
    )
    args = parser.parse_args()

    args.output_dir = add_detail_suffix(args.output_dir, args.detail)

    ground_truth_map = load_ground_truth_map(args.ground_truth_path)
    done_count = analyze_rollouts(
        results_dir=args.results_dir,
        ground_truth_map=ground_truth_map,
        db_root_path=args.db_root_path,
        timeout_s=args.timeout,
        num_workers=max(1, args.num_workers),
        output_dir=args.output_dir,
        detail=args.detail,
        max_pkl=args.max_pkl,
    )

    print(f"Saved per-point files to: {args.output_dir}")
    print(f"Total samples: {done_count}")


if __name__ == "__main__":
    main()
