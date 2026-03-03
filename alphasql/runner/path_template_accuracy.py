import argparse
import glob
import json
import pickle
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from func_timeout import FunctionTimedOut, func_timeout
from tqdm import tqdm

def _parse_ground_truth(ground_truth_path: str) -> Dict[int, Tuple[str, str]]:
    gt_map: Dict[int, Tuple[str, str]] = {}
    with open(ground_truth_path, "r") as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        sql, db_id = line.split("\t")
        gt_map[idx] = (sql, db_id)
    return gt_map


def _parse_difficulties(diff_json_path: str) -> Dict[int, str]:
    with open(diff_json_path, "r") as f:
        contents = json.load(f)
    difficulty_by_qid: Dict[int, str] = {}
    for idx, content in enumerate(contents):
        difficulty_by_qid[idx] = content.get("difficulty", "unknown")
    return difficulty_by_qid


def _count_difficulty_totals(difficulty_by_qid: Dict[int, str]) -> Dict[str, int]:
    totals = {"simple": 0, "moderate": 0, "challenging": 0, "unknown": 0}
    for difficulty in difficulty_by_qid.values():
        if difficulty in totals:
            totals[difficulty] += 1
        else:
            totals["unknown"] += 1
    return totals


def _path_signature(path_nodes: List[object]) -> str:
    return "->".join([node.node_type.value for node in path_nodes])


def _execute_sql(predicted_sql: str, ground_truth: str, db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    conn.close()
    return 1 if set(predicted_res) == set(ground_truth_res) else 0


def _execute_model(predicted_sql: str, ground_truth: str, db_path: str, meta_time_out: float) -> int:
    try:
        return func_timeout(meta_time_out, _execute_sql, args=(predicted_sql, ground_truth, db_path))
    except FunctionTimedOut:
        return False
    except Exception:
        return False


def _process_one_file(
    results_file_path: str,
    db_root_dir: str,
    gt_map: Dict[int, Tuple[str, str]],
    difficulty_by_qid: Dict[int, str],
    meta_time_out: float,
) -> Dict[str, Dict[str, int]]:
    question_id = int(Path(results_file_path).stem)
    if question_id not in gt_map:
        return {}

    gt_sql, gt_db_id = gt_map[question_id]
    db_path = str(Path(db_root_dir) / gt_db_id / f"{gt_db_id}.sqlite")
    difficulty = difficulty_by_qid.get(question_id, "unknown")

    with open(results_file_path, "rb") as f:
        reasoning_paths = pickle.load(f)

    if not reasoning_paths:
        return {}

    per_signature_best_correct: Dict[str, int] = {}

    for path_nodes in reasoning_paths:
        if not path_nodes:
            continue
        end_node = path_nodes[-1]
        pred_sql = getattr(end_node, "final_sql_query", None)
        if not pred_sql:
            continue

        signature = _path_signature(path_nodes)
        is_correct = _execute_model(pred_sql, gt_sql, db_path, meta_time_out)
        if signature not in per_signature_best_correct:
            per_signature_best_correct[signature] = is_correct
        else:
            per_signature_best_correct[signature] = max(per_signature_best_correct[signature], is_correct)

    aggregated: Dict[str, Dict[str, int]] = {}
    for signature, correct in per_signature_best_correct.items():
        stat = {
            "appear_questions": 1,
            "correct_questions": correct,
            "appear_simple": 0,
            "appear_moderate": 0,
            "appear_challenging": 0,
            "correct_simple": 0,
            "correct_moderate": 0,
            "correct_challenging": 0,
        }
        if difficulty == "simple":
            stat["appear_simple"] = 1
            stat["correct_simple"] = correct
        elif difficulty == "moderate":
            stat["appear_moderate"] = 1
            stat["correct_moderate"] = correct
        elif difficulty == "challenging":
            stat["appear_challenging"] = 1
            stat["correct_challenging"] = correct

        aggregated[signature] = {
            **stat,
        }
    return aggregated


def main(args):
    gt_map = _parse_ground_truth(args.ground_truth_path)
    difficulty_by_qid = _parse_difficulties(args.diff_json_path)
    total_questions = len(gt_map)
    difficulty_totals = _count_difficulty_totals(difficulty_by_qid)

    result_paths = sorted(glob.glob(str(Path(args.results_dir) / "*.pkl")))
    if not result_paths:
        raise ValueError(f"No pkl files found in {args.results_dir}")

    merged_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "appear_questions": 0,
            "correct_questions": 0,
            "appear_simple": 0,
            "appear_moderate": 0,
            "appear_challenging": 0,
            "correct_simple": 0,
            "correct_moderate": 0,
            "correct_challenging": 0,
        }
    )

    with ProcessPoolExecutor(max_workers=args.process_num) as executor:
        futures = [
            executor.submit(
                _process_one_file,
                path,
                args.db_root_dir,
                gt_map,
                difficulty_by_qid,
                args.meta_time_out,
            )
            for path in result_paths
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing path templates"):
            one_file_stats = future.result()
            for signature, stat in one_file_stats.items():
                for key, value in stat.items():
                    merged_stats[signature][key] += value

    rows = []
    for signature, stat in merged_stats.items():
        appear_questions = stat["appear_questions"]
        correct_questions = stat["correct_questions"]

        correct_simple = stat["correct_simple"]
        correct_moderate = stat["correct_moderate"]
        correct_challenging = stat["correct_challenging"]

        simple_total = difficulty_totals["simple"]
        moderate_total = difficulty_totals["moderate"]
        challenging_total = difficulty_totals["challenging"]

        conditional_acc = correct_questions / appear_questions if appear_questions > 0 else 0.0
        simple_conditional_acc = correct_simple / stat["appear_simple"] if stat["appear_simple"] > 0 else 0.0
        moderate_conditional_acc = correct_moderate / stat["appear_moderate"] if stat["appear_moderate"] > 0 else 0.0
        challenging_conditional_acc = (
            correct_challenging / stat["appear_challenging"] if stat["appear_challenging"] > 0 else 0.0
        )
        coverage = appear_questions / total_questions if total_questions > 0 else 0.0
        rows.append(
            {
                "path_template": signature,
                "correct_questions": correct_questions,
                "appear_questions": appear_questions,
                "total_questions": total_questions,
                "conditional_accuracy": conditional_acc,
                "simple_accuracy": simple_conditional_acc,
                "moderate_accuracy": moderate_conditional_acc,
                "challenging_accuracy": challenging_conditional_acc,
                "coverage": coverage,
                "correct_simple": correct_simple,
                "correct_moderate": correct_moderate,
                "correct_challenging": correct_challenging,
                "simple_total": simple_total,
                "moderate_total": moderate_total,
                "challenging_total": challenging_total,
                "appear_simple": stat["appear_simple"],
                "appear_moderate": stat["appear_moderate"],
                "appear_challenging": stat["appear_challenging"],
            }
        )

    rows.sort(
        key=lambda x: (
            x["conditional_accuracy"],
            x["correct_questions"],
            x["simple_accuracy"],
            x["moderate_accuracy"],
            x["challenging_accuracy"],
        ),
        reverse=True,
    )

    output = {
        "results_dir": args.results_dir,
        "ground_truth_path": args.ground_truth_path,
        "diff_json_path": args.diff_json_path,
        "db_root_dir": args.db_root_dir,
        "difficulty_totals": difficulty_totals,
        "num_templates": len(rows),
        "templates": rows,
    }

    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved ranked template accuracy to {args.output_path}")
    if rows:
        print("Top templates by accuracy (denominator = appear_questions):")
        for idx, row in enumerate(rows[: min(args.top_k, len(rows))], start=1):
            print(
                f"{idx:>2}. acc={row['conditional_accuracy']:.4f}, "
                f"simple={row['simple_accuracy']:.4f}, "
                f"moderate={row['moderate_accuracy']:.4f}, "
                f"challenging={row['challenging_accuracy']:.4f}, "
                f"cond_acc={row['conditional_accuracy']:.4f}, "
                f"coverage={row['coverage']:.4f}, "
                f"correct={row['correct_questions']}/{row['appear_questions']} | {row['path_template']}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-path-template accuracy ranked by appear_questions-based accuracy")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing per-question pkl files")
    parser.add_argument("--ground_truth_path", type=str, required=True, help="Path to ground-truth SQL file (e.g., dev.sql)")
    parser.add_argument("--diff_json_path", type=str, required=True, help="Path to difficulty json (e.g., dev.json)")
    parser.add_argument("--db_root_dir", type=str, required=True, help="Database root directory")
    parser.add_argument("--output_path", type=str, required=True, help="Output json file path")
    parser.add_argument("--process_num", type=int, default=16, help="Number of worker processes")
    parser.add_argument("--meta_time_out", type=float, default=30.0, help="Per SQL evaluation timeout in seconds")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top templates printed to stdout")
    args = parser.parse_args()
    main(args)
