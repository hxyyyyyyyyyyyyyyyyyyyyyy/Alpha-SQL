#!/usr/bin/env python3
"""Collect failed data points after SQL selection and copy analysis files.

Failure criterion is intentionally aligned with alphasql.runner.evaluation:
- Execute predicted SQL and golden SQL on the same sqlite database.
- Compare set(predicted_res) == set(ground_truth_res).
- Timeout / execution errors are treated as incorrect.
"""

import argparse
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

from func_timeout import FunctionTimedOut, func_timeout


def execute_sql_same_standard(predicted_sql: str, ground_truth: str, db_path: Path) -> int:
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


def judge_sql(predicted_sql: str, ground_truth_sql: str, db_path: Path, timeout_s: float) -> Tuple[int, str]:
    if not isinstance(predicted_sql, str) or not predicted_sql.strip():
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


def load_ground_truth_lines(ground_truth_path: Path) -> List[str]:
    with ground_truth_path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def parse_pred_sql(pred_item: object) -> str:
    if isinstance(pred_item, str) and "\t----- bird -----\t" in pred_item:
        return pred_item.split("\t----- bird -----\t", 1)[0]
    return ""


def collect_fail_points(
    pred_sql_path: Path,
    ground_truth_path: Path,
    db_root_dir: Path,
    analysis_dir: Path,
    fail_dir: Path,
    timeout_s: float,
) -> Dict[str, object]:
    with pred_sql_path.open("r", encoding="utf-8") as f:
        pred_data = json.load(f)

    gt_lines = load_ground_truth_lines(ground_truth_path)

    fail_items: List[Dict[str, object]] = []
    copied = 0
    missing_analysis_files: List[str] = []

    for qid_str, pred_item in sorted(pred_data.items(), key=lambda x: int(x[0])):
        qid = int(qid_str)
        if qid < 0 or qid >= len(gt_lines):
            continue

        parts = gt_lines[qid].split("\t")
        if len(parts) < 2:
            continue

        golden_sql, db_name = parts[0], parts[1]
        db_path = db_root_dir / db_name / f"{db_name}.sqlite"
        pred_sql = parse_pred_sql(pred_item)

        result, error_msg = judge_sql(pred_sql, golden_sql, db_path, timeout_s)
        if result == 1:
            continue

        fail_items.append(
            {
                "question_id": qid,
                "result": result,
                "error_msg": error_msg,
            }
        )

        src_analysis_file = analysis_dir / f"{qid}.json"
        dst_analysis_file = fail_dir / f"{qid}.json"
        if src_analysis_file.exists():
            shutil.copy2(src_analysis_file, dst_analysis_file)
            copied += 1
        else:
            missing_analysis_files.append(str(src_analysis_file))

    return {
        "pred_sql_path": str(pred_sql_path),
        "total_pred_items": len(pred_data),
        "fail_count": len(fail_items),
        "copied_analysis_files": copied,
        "missing_analysis_files": missing_analysis_files,
        "fail_items": fail_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect failed samples and copy analysis files")
    parser.add_argument("--pred_sql_path", type=Path, required=True)
    parser.add_argument("--ground_truth_path", type=Path, required=True)
    parser.add_argument("--db_root_dir", type=Path, required=True)
    parser.add_argument("--analysis_dir", type=Path, required=True)
    parser.add_argument("--fail_dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    args.fail_dir.mkdir(parents=True, exist_ok=True)
    for old_json in args.fail_dir.glob("[0-9]*.json"):
        old_json.unlink()

    summary = collect_fail_points(
        pred_sql_path=args.pred_sql_path,
        ground_truth_path=args.ground_truth_path,
        db_root_dir=args.db_root_dir,
        analysis_dir=args.analysis_dir,
        fail_dir=args.fail_dir,
        timeout_s=args.timeout,
    )

    summary_path = args.fail_dir / "fail_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print(
        f"[FAIL COPY] pred={args.pred_sql_path}, fail_count={summary['fail_count']}, "
        f"copied={summary['copied_analysis_files']}, fail_dir={args.fail_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
