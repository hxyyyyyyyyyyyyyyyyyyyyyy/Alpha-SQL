#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm
from transformers import AutoTokenizer

from alphasql.llm_call.prompt_factory import get_prompt

try:
    from alphasql.database.utils import load_database_schema_dict as load_database_schema_dict_repo
    LOAD_SCHEMA_SOURCE = "alphasql.database.utils.load_database_schema_dict"
except Exception:
    load_database_schema_dict_repo = None
    LOAD_SCHEMA_SOURCE = "fallback_local_repo_compatible_loader"


@dataclass
class TaskLite:
    question_id: int
    db_id: str
    question: str
    evidence: str


@dataclass
class FieldExampleInfo:
    db_id: str
    table_name: str
    column_name: str
    column_type: str
    examples: List[str]
    max_example_char_len: int
    example_char_lens: List[int]
    schema_context_needed: bool


def _execute_sql_without_timeout(db_path: Path, query: str) -> List[Tuple[Any, ...]]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.text_factory = lambda x: str(x, "utf-8", errors="replace")
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()


def _normalize_description_string(description: str) -> str:
    description = description.replace("\r", "").replace("\n", "").replace("commonsense evidence:", "").strip()
    while "  " in description:
        description = description.replace("  ", " ")
    return description


def _load_database_description_fallback(db_id: str, database_root_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    db_description_dir = Path(database_root_dir) / db_id / "database_description"
    if not db_description_dir.exists():
        return {}
    database_description: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for csv_file in db_description_dir.glob("*.csv"):
        table_name_lower = csv_file.stem.lower().strip()
        table_description: Dict[str, Dict[str, Any]] = {}

        rows = None
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                with open(csv_file, "r", encoding=enc, newline="") as f:
                    rows = list(csv.DictReader(f))
                break
            except Exception:
                continue
        if rows is None:
            continue

        for row in rows:
            original_column_name = (row.get("original_column_name") or "").strip()
            if original_column_name == "":
                continue
            original_column_name_lower = original_column_name.lower()
            expanded_column_name = (row.get("column_name") or "").strip()
            column_description = _normalize_description_string((row.get("column_description") or ""))
            data_format = (row.get("data_format") or "").strip()
            value_description = _normalize_description_string((row.get("value_description") or ""))
            if value_description.lower().startswith("not useful"):
                value_description = value_description[len("not useful") :].strip()
            table_description[original_column_name_lower] = {
                "original_column_name_lower": original_column_name_lower,
                "expanded_column_name": expanded_column_name,
                "column_description": column_description,
                "data_format": data_format,
                "value_description": value_description,
            }
        database_description[table_name_lower] = table_description
    return database_description


def load_database_schema_dict_fallback(db_id: str, database_root_dir: str) -> Dict[str, Any]:
    db_path = Path(database_root_dir) / db_id / f"{db_id}.sqlite"
    table_names_rows = _execute_sql_without_timeout(
        db_path,
        "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';",
    )
    table_names = [row[0].strip() for row in table_names_rows]
    database_description = _load_database_description_fallback(db_id, database_root_dir)

    database_schema_dict: Dict[str, Any] = {
        "db_id": db_id,
        "db_directory": Path(database_root_dir) / db_id,
        "tables": {},
    }

    for table_name in table_names:
        table_schema_dict: Dict[str, Any] = {
            "table_name": table_name,
            "columns": {},
        }

        table_info = _execute_sql_without_timeout(db_path, f"PRAGMA table_info(`{table_name}`);")
        primary_keys = [row[1].strip() for row in table_info if row[5] != 0]

        foreign_keys_list = _execute_sql_without_timeout(db_path, f"PRAGMA foreign_key_list(`{table_name}`);")
        foreign_keys: List[Tuple[str, str, str, str]] = []
        for foreign_key in foreign_keys_list:
            source_table_name = table_name.strip()
            source_column_name = foreign_key[3].strip()
            target_table_name = foreign_key[2].strip()
            target_column_name = foreign_key[4].strip() if foreign_key[4] is not None else ""
            if target_column_name == "":
                target_table_info = _execute_sql_without_timeout(db_path, f"PRAGMA table_info(`{target_table_name}`);")
                target_pks = [row[1].strip() for row in target_table_info if row[5] != 0]
                target_column_name = target_pks[0] if len(target_pks) > 0 else ""
            if target_column_name != "":
                foreign_keys.append((source_table_name, source_column_name, target_table_name, target_column_name))

        for row in table_info:
            column_name = row[1].strip()
            column_type = row[2].strip()
            column_schema_dict: Dict[str, Any] = {
                "original_column_name": column_name,
                "column_type": column_type,
                "foreign_keys": [],
                "referenced_by": [],
                "primary_key": column_name.lower() in [x.lower() for x in primary_keys],
                "expanded_column_name": "",
                "column_description": "",
                "value_description": "",
            }

            for source_table_name, source_column_name, target_table_name, target_column_name in foreign_keys:
                if source_table_name.lower() == table_name.lower() and source_column_name.lower() == column_name.lower():
                    column_schema_dict["foreign_keys"].append((target_table_name, target_column_name))
                if target_table_name.lower() == table_name.lower() and target_column_name.lower() == column_name.lower():
                    column_schema_dict["referenced_by"].append((source_table_name, source_column_name))

            if table_name.lower() in database_description and column_name.lower() in database_description[table_name.lower()]:
                desc = database_description[table_name.lower()][column_name.lower()]
                column_schema_dict["expanded_column_name"] = desc.get("expanded_column_name", "")
                column_schema_dict["column_description"] = desc.get("column_description", "")
                column_schema_dict["value_description"] = desc.get("value_description", "")

            if column_type.upper() != "BLOB":
                examples_rows = _execute_sql_without_timeout(
                    db_path,
                    f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL AND `{column_name}` != '' LIMIT 3;",
                )
                column_schema_dict["value_examples"] = [example[0] for example in examples_rows]
            else:
                column_schema_dict["value_examples"] = []

            table_schema_dict["columns"][column_name] = column_schema_dict

        database_schema_dict["tables"][table_name] = table_schema_dict

    return database_schema_dict


def load_database_schema_dict_compatible(db_id: str, database_root_dir: str) -> Dict[str, Any]:
    if load_database_schema_dict_repo is not None:
        return load_database_schema_dict_repo(db_id=db_id, database_root_dir=database_root_dir)
    return load_database_schema_dict_fallback(db_id=db_id, database_root_dir=database_root_dir)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    return value


def _percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] * (hi - idx) + sorted_values[hi] * (idx - lo))


def summarize_numeric(values: List[int]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "min": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "mean": 0,
        }
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "min": int(sorted_values[0]),
        "p50": round(_percentile(sorted_values, 0.50), 2),
        "p90": round(_percentile(sorted_values, 0.90), 2),
        "p95": round(_percentile(sorted_values, 0.95), 2),
        "p99": round(_percentile(sorted_values, 0.99), 2),
        "max": int(sorted_values[-1]),
        "mean": round(mean(values), 2),
    }


def build_table_ddl_statement_with_limit(
    table_schema_dict: Dict[str, Any],
    max_example_length: int,
    add_expanded_column_name: bool = True,
    add_column_description: bool = True,
    add_value_description: bool = True,
    add_value_examples: bool = True,
) -> str:
    statement = f"CREATE TABLE `{table_schema_dict['table_name']}` (\n"
    foreign_keys = []
    primary_keys = []
    for column_name, column_schema in table_schema_dict["columns"].items():
        column_statement = f"\t`{column_name}` {column_schema['column_type']},"
        comment_parts = []
        expanded_name = column_schema.get("expanded_column_name", "")
        column_description = column_schema.get("column_description", "")
        value_description = column_schema.get("value_description", "")
        value_examples = column_schema.get("value_examples", [])

        if add_expanded_column_name and expanded_name.strip() != "" and expanded_name.strip().lower() != column_name.lower():
            comment_parts.append(f"Column Meaning: {expanded_name}")
        if (
            add_column_description
            and column_description.strip() != ""
            and column_description.strip().lower() != column_name.lower()
            and column_description.strip().lower() != expanded_name.strip().lower()
        ):
            comment_parts.append(f"Column Description: {column_description}")
        if add_value_description and value_description.strip() != "":
            comment_parts.append(f"Value Description: {value_description}")
        if add_value_examples and len(value_examples) > 0 and str(column_schema["column_type"]).upper() == "TEXT":
            if all(len(str(v)) <= max_example_length for v in value_examples):
                comment_parts.append(f"Value Examples: {', '.join([f'`{value}`' for value in value_examples])}")

        if len(comment_parts) > 0:
            column_statement += f" -- {' | '.join(comment_parts)}"
        statement += column_statement + "\n"

        if column_schema.get("primary_key", False):
            primary_keys.append(column_name)
        for foreign_key in column_schema.get("foreign_keys", []):
            foreign_keys.append((column_name, *foreign_key))

    statement += "\tPRIMARY KEY (" + ", ".join([f"`{primary_key}`" for primary_key in primary_keys]) + "),\n"
    for source_column_name, target_table_name, target_column_name in foreign_keys:
        statement += f"\tFOREIGN KEY (`{source_column_name}`) REFERENCES `{target_table_name}`(`{target_column_name}`),\n"
    if statement[-2:] == ",\n":
        statement = statement[:-2]
    statement += "\n);"
    return statement


def build_schema_context_with_limit(database_schema_dict: Dict[str, Any], max_example_length: int) -> str:
    table_ddls = []
    for _, table_schema_dict in database_schema_dict["tables"].items():
        table_ddls.append(
            build_table_ddl_statement_with_limit(
                table_schema_dict=table_schema_dict,
                max_example_length=max_example_length,
                add_expanded_column_name=True,
                add_column_description=True,
                add_value_description=True,
                add_value_examples=True,
            )
        )
    return "\n".join(table_ddls)


def collect_field_examples_info(all_db_schema_dict: Dict[str, Dict[str, Any]]) -> List[FieldExampleInfo]:
    records: List[FieldExampleInfo] = []
    for db_id, db_schema_dict in all_db_schema_dict.items():
        for table_name, table_schema_dict in db_schema_dict["tables"].items():
            for column_name, column_schema_dict in table_schema_dict["columns"].items():
                column_type = str(column_schema_dict.get("column_type", "")).upper()
                examples = [str(x) for x in column_schema_dict.get("value_examples", [])]
                example_lens = [len(x) for x in examples]
                max_len = max(example_lens) if example_lens else 0
                schema_context_needed = column_type == "TEXT" and len(examples) > 0
                records.append(
                    FieldExampleInfo(
                        db_id=db_id,
                        table_name=table_name,
                        column_name=column_name,
                        column_type=column_type,
                        examples=examples,
                        max_example_char_len=max_len,
                        example_char_lens=example_lens,
                        schema_context_needed=schema_context_needed,
                    )
                )
    return records


def count_rejected_fields(field_infos: List[FieldExampleInfo], threshold: int) -> int:
    rejected = 0
    for info in field_infos:
        if not info.schema_context_needed:
            continue
        if info.max_example_char_len > threshold:
            rejected += 1
    return rejected


def token_count_plain(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def token_count_chat_user(tokenizer: AutoTokenizer, prompt: str) -> int:
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def split_sql_generation_prompt_parts(schema_context: str, question: str, hint: str) -> Dict[str, str]:
    prompt = get_prompt(
        "sql_generation",
        {"SCHEMA_CONTEXT": schema_context, "QUESTION": question, "HINT": hint},
    )
    schema_marker = "**************************\n【Table creation statements】\n"
    question_marker = "\n\n**************************\n【Question】\nQuestion: \n"
    hint_marker = "\n\nHint:\n"
    end_marker = "\n\n**************************\n\nOnly output xml format (starting with ```xml and ending with ```) as your response."

    s_idx = prompt.find(schema_marker)
    q_idx = prompt.find(question_marker)
    h_idx = prompt.find(hint_marker)
    e_idx = prompt.rfind(end_marker)

    if min(s_idx, q_idx, h_idx, e_idx) < 0:
        raise RuntimeError("Failed to split sql_generation prompt into parts. Template markers not found.")

    schema_content_start = s_idx + len(schema_marker)
    question_content_start = q_idx + len(question_marker)
    hint_content_start = h_idx + len(hint_marker)

    return {
        "prompt": prompt,
        "prefix_instruction": prompt[:schema_content_start],
        "schema_context": prompt[schema_content_start:q_idx],
        "question_wrapper": prompt[q_idx:question_content_start],
        "question_text": prompt[question_content_start:h_idx],
        "hint_wrapper": prompt[h_idx:hint_content_start],
        "hint_text": prompt[hint_content_start:e_idx],
        "suffix_instruction": prompt[e_idx:],
    }


def evaluate_threshold(
    threshold: int,
    tasks: List[TaskLite],
    schema_cache: Dict[Tuple[str, int], str],
    all_db_schema_dict: Dict[str, Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_eval_tasks: int,
) -> Dict[str, Any]:
    if max_eval_tasks > 0:
        eval_tasks = tasks[:max_eval_tasks]
    else:
        eval_tasks = tasks

    part_stats: Dict[str, List[int]] = {
        "prefix_instruction": [],
        "schema_context": [],
        "question_wrapper": [],
        "question_text": [],
        "hint_wrapper": [],
        "hint_text": [],
        "suffix_instruction": [],
        "plain_prompt_total": [],
        "chat_prompt_total": [],
    }

    max_prompt = -1
    max_prompt_item: Dict[str, Any] = {}

    for task in tqdm(eval_tasks, desc=f"Evaluate threshold={threshold}", leave=False):
        cache_key = (task.db_id, threshold)
        if cache_key not in schema_cache:
            schema_cache[cache_key] = build_schema_context_with_limit(all_db_schema_dict[task.db_id], threshold)
        schema_context = schema_cache[cache_key]

        parts = split_sql_generation_prompt_parts(
            schema_context=schema_context,
            question=task.question,
            hint=task.evidence,
        )
        part_token_counts = {k: token_count_plain(tokenizer, v) for k, v in parts.items() if k != "prompt"}

        plain_total = token_count_plain(tokenizer, parts["prompt"])
        chat_total = token_count_chat_user(tokenizer, parts["prompt"])

        for k in [
            "prefix_instruction",
            "schema_context",
            "question_wrapper",
            "question_text",
            "hint_wrapper",
            "hint_text",
            "suffix_instruction",
        ]:
            part_stats[k].append(part_token_counts[k])
        part_stats["plain_prompt_total"].append(plain_total)
        part_stats["chat_prompt_total"].append(chat_total)

        if chat_total > max_prompt:
            max_prompt = chat_total
            max_prompt_item = {
                "question_id": task.question_id,
                "db_id": task.db_id,
                "chat_prompt_tokens": chat_total,
                "plain_prompt_tokens": plain_total,
                "part_tokens": part_token_counts,
                "question": task.question,
            }

    summary = {k: summarize_numeric(v) for k, v in part_stats.items()}
    return {
        "threshold": threshold,
        "task_count": len(eval_tasks),
        "token_summary": summary,
        "max_prompt_item": max_prompt_item,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze example-length threshold vs Qwen context limit.")
    parser.add_argument("--dev-json", type=str, default="data/bird/dev/dev.json")
    parser.add_argument("--db-root", type=str, default="data/bird/dev/dev_databases")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--context-limit", type=int, default=32768)
    parser.add_argument("--reserve-completion", type=int, default=4096)
    parser.add_argument("--baseline-threshold", type=int, default=1000)
    parser.add_argument("--max-eval-tasks", type=int, default=0, help="0 means evaluate all tasks")
    parser.add_argument("--output", type=str, default="tools/example_length_context_report.json")
    args = parser.parse_args()

    dev_json_path = Path(args.dev_json)
    db_root = Path(args.db_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_data = json.loads(dev_json_path.read_text(encoding="utf-8"))
    tasks = [
        TaskLite(
            question_id=item.get("question_id", i),
            db_id=item["db_id"],
            question=item["question"],
            evidence=item.get("evidence", ""),
        )
        for i, item in enumerate(raw_data)
    ]
    all_db_ids = sorted({task.db_id for task in tasks})

    print(f"Loaded {len(tasks)} tasks from {dev_json_path}")
    print(f"Found {len(all_db_ids)} unique databases")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading database schemas via repository utilities...")
    all_db_schema_dict: Dict[str, Dict[str, Any]] = {}
    for db_id in tqdm(all_db_ids, desc="Load db schemas"):
        all_db_schema_dict[db_id] = load_database_schema_dict_compatible(db_id=db_id, database_root_dir=str(db_root))

    print("Collecting value example length distributions...")
    field_infos = collect_field_examples_info(all_db_schema_dict)
    eligible_field_infos = [info for info in field_infos if info.schema_context_needed]

    all_example_lens = [l for info in eligible_field_infos for l in info.example_char_lens]
    field_max_lens = [info.max_example_char_len for info in eligible_field_infos]

    print(f"Total fields: {len(field_infos)}, TEXT fields with examples: {len(eligible_field_infos)}")

    unique_candidate_thresholds = sorted(set([0, args.baseline_threshold] + field_max_lens))
    prompt_budget = args.context_limit - args.reserve_completion

    schema_cache: Dict[Tuple[str, int], str] = {}

    print("Evaluating baseline threshold token usage...")
    baseline_eval = evaluate_threshold(
        threshold=args.baseline_threshold,
        tasks=tasks,
        schema_cache=schema_cache,
        all_db_schema_dict=all_db_schema_dict,
        tokenizer=tokenizer,
        max_eval_tasks=args.max_eval_tasks,
    )

    print("Searching safe max_example_length threshold...")
    lo, hi = 0, len(unique_candidate_thresholds) - 1
    best_idx = -1
    best_eval: Dict[str, Any] = {}

    while lo <= hi:
        mid = (lo + hi) // 2
        t = unique_candidate_thresholds[mid]
        result = evaluate_threshold(
            threshold=t,
            tasks=tasks,
            schema_cache=schema_cache,
            all_db_schema_dict=all_db_schema_dict,
            tokenizer=tokenizer,
            max_eval_tasks=args.max_eval_tasks,
        )
        max_chat_tokens = result["token_summary"]["chat_prompt_total"]["max"]
        if max_chat_tokens <= prompt_budget:
            best_idx = mid
            best_eval = result
            lo = mid + 1
        else:
            hi = mid - 1

    if best_idx == -1:
        safe_threshold = 0
        safe_eval = evaluate_threshold(
            threshold=0,
            tasks=tasks,
            schema_cache=schema_cache,
            all_db_schema_dict=all_db_schema_dict,
            tokenizer=tokenizer,
            max_eval_tasks=args.max_eval_tasks,
        )
    else:
        safe_threshold = unique_candidate_thresholds[best_idx]
        safe_eval = best_eval

    print("Evaluating threshold under full context limit (without reserved completion)...")
    lo2, hi2 = 0, len(unique_candidate_thresholds) - 1
    best_idx_full = -1
    best_eval_full: Dict[str, Any] = {}
    while lo2 <= hi2:
        mid = (lo2 + hi2) // 2
        t = unique_candidate_thresholds[mid]
        result = evaluate_threshold(
            threshold=t,
            tasks=tasks,
            schema_cache=schema_cache,
            all_db_schema_dict=all_db_schema_dict,
            tokenizer=tokenizer,
            max_eval_tasks=args.max_eval_tasks,
        )
        max_chat_tokens = result["token_summary"]["chat_prompt_total"]["max"]
        if max_chat_tokens <= args.context_limit:
            best_idx_full = mid
            best_eval_full = result
            lo2 = mid + 1
        else:
            hi2 = mid - 1

    if best_idx_full == -1:
        safe_threshold_full = 0
        safe_eval_full = evaluate_threshold(
            threshold=0,
            tasks=tasks,
            schema_cache=schema_cache,
            all_db_schema_dict=all_db_schema_dict,
            tokenizer=tokenizer,
            max_eval_tasks=args.max_eval_tasks,
        )
    else:
        safe_threshold_full = unique_candidate_thresholds[best_idx_full]
        safe_eval_full = best_eval_full

    rejected_at_safe = count_rejected_fields(eligible_field_infos, safe_threshold)
    rejected_at_baseline = count_rejected_fields(eligible_field_infos, args.baseline_threshold)
    rejected_at_full = count_rejected_fields(eligible_field_infos, safe_threshold_full)

    top_longest_fields = sorted(
        eligible_field_infos,
        key=lambda x: x.max_example_char_len,
        reverse=True,
    )[:30]

    report = {
        "meta": {
            "dev_json": str(dev_json_path),
            "db_root": str(db_root),
            "model": args.model,
            "schema_loader": LOAD_SCHEMA_SOURCE,
            "context_limit": args.context_limit,
            "reserve_completion": args.reserve_completion,
            "prompt_budget": prompt_budget,
            "baseline_threshold": args.baseline_threshold,
            "evaluated_task_count": safe_eval["task_count"],
            "total_task_count": len(tasks),
            "total_db_count": len(all_db_ids),
        },
        "distribution": {
            "all_example_char_lens": summarize_numeric(all_example_lens),
            "field_max_example_char_lens": summarize_numeric(field_max_lens),
            "total_fields": len(field_infos),
            "text_fields_with_examples": len(eligible_field_infos),
            "top_longest_fields": [
                {
                    "db_id": info.db_id,
                    "table_name": info.table_name,
                    "column_name": info.column_name,
                    "max_example_char_len": info.max_example_char_len,
                    "example_char_lens": info.example_char_lens,
                    "examples_preview": [x[:120] for x in info.examples],
                }
                for info in top_longest_fields
            ],
        },
        "thresholds": {
            "safe_with_reserve": {
                "max_example_length": safe_threshold,
                "max_chat_prompt_tokens": safe_eval["token_summary"]["chat_prompt_total"]["max"],
                "rejected_field_count": rejected_at_safe,
                "token_breakdown_summary": safe_eval["token_summary"],
                "worst_case_task": safe_eval["max_prompt_item"],
            },
            "safe_without_reserve": {
                "max_example_length": safe_threshold_full,
                "max_chat_prompt_tokens": safe_eval_full["token_summary"]["chat_prompt_total"]["max"],
                "rejected_field_count": rejected_at_full,
                "token_breakdown_summary": safe_eval_full["token_summary"],
                "worst_case_task": safe_eval_full["max_prompt_item"],
            },
            "baseline": {
                "max_example_length": args.baseline_threshold,
                "max_chat_prompt_tokens": baseline_eval["token_summary"]["chat_prompt_total"]["max"],
                "rejected_field_count": rejected_at_baseline,
                "token_breakdown_summary": baseline_eval["token_summary"],
                "worst_case_task": baseline_eval["max_prompt_item"],
            },
        },
    }

    output_path.write_text(json.dumps(_to_serializable(report), ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 120)
    print("Safe threshold with reserve:")
    print(
        f"max_example_length={safe_threshold}, max_chat_prompt_tokens={report['thresholds']['safe_with_reserve']['max_chat_prompt_tokens']}, "
        f"rejected_fields={rejected_at_safe}"
    )
    print("Safe threshold without reserve:")
    print(
        f"max_example_length={safe_threshold_full}, max_chat_prompt_tokens={report['thresholds']['safe_without_reserve']['max_chat_prompt_tokens']}, "
        f"rejected_fields={rejected_at_full}"
    )
    print("Baseline threshold:")
    print(
        f"max_example_length={args.baseline_threshold}, max_chat_prompt_tokens={report['thresholds']['baseline']['max_chat_prompt_tokens']}, "
        f"rejected_fields={rejected_at_baseline}"
    )
    print(f"Report saved to {output_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()