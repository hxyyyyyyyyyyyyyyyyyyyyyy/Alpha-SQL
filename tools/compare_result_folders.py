#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Ensure project root is importable when running as `python script/compare_result_folders.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _normalize_name(value: Any, lowercase: bool = True) -> str:
    if value is None:
        return "unknown"
    if hasattr(value, "value"):
        text = str(value.value)
    else:
        text = str(value)
    if "." in text:
        text = text.split(".")[-1]
    return text.lower() if lowercase else text


def _safe_sort_counter(counter: Counter) -> List[Tuple[str, int]]:
    return sorted(counter.items(), key=lambda x: (-x[1], x[0]))


def analyze_folder(folder_path: Path, top_k_json: int = 20) -> Dict[str, Any]:
    pkl_files = sorted(folder_path.glob("*.pkl"))

    failed_files: List[str] = []
    error_counter: Counter = Counter()

    loaded_files = 0
    total_paths = 0
    total_nodes = 0

    node_counter: Counter = Counter()
    action_counter: Counter = Counter()
    node_path_counter: Counter = Counter()
    action_path_counter: Counter = Counter()
    position_node_counter: Dict[int, Counter] = defaultdict(Counter)

    for pkl_file in pkl_files:
        try:
            with pkl_file.open("rb") as f:
                data = pickle.load(f)
        except Exception as e:
            failed_files.append(pkl_file.name)
            error_counter[e.__class__.__name__] += 1
            continue

        if not isinstance(data, list):
            failed_files.append(pkl_file.name)
            error_counter["NonListPickleContent"] += 1
            continue

        loaded_files += 1

        for path_nodes in data:
            if not isinstance(path_nodes, list) or not path_nodes:
                continue

            total_paths += 1
            total_nodes += len(path_nodes)

            node_seq: List[str] = []
            action_seq: List[str] = []

            for step, node in enumerate(path_nodes):
                node_name = _normalize_name(getattr(node, "node_type", None), lowercase=True)
                node_counter[node_name] += 1
                position_node_counter[step][node_name] += 1
                node_seq.append(node_name)

                parent_action = getattr(node, "parent_action", None)
                if parent_action is not None:
                    action_name = parent_action.__class__.__name__
                    action_counter[action_name] += 1
                    action_seq.append(action_name)

            node_path_counter[" -> ".join(node_seq)] += 1
            if action_seq:
                action_path_counter[" -> ".join(action_seq)] += 1

    avg_nodes_per_path = (total_nodes / total_paths) if total_paths else 0.0

    return {
        "folder": str(folder_path),
        "pkl_files": len(pkl_files),
        "failed_files": failed_files,
        "error_counter": dict(error_counter),
        "loaded_files": loaded_files,
        "total_paths": total_paths,
        "total_nodes": total_nodes,
        "avg_nodes_per_path": avg_nodes_per_path,
        "node_counter": dict(node_counter),
        "action_counter": dict(action_counter),
        "top_node_paths": _safe_sort_counter(node_path_counter)[:top_k_json],
        "top_action_paths": _safe_sort_counter(action_path_counter)[:top_k_json],
        "position_node_counter": {
            str(step): dict(counter)
            for step, counter in sorted(position_node_counter.items(), key=lambda x: x[0])
        },
    }


def format_count_pct(count: int, total: int, digits: int = 2) -> str:
    if total <= 0:
        return f"{count} (0.00%)"
    return f"{count} ({count / total * 100:.{digits}f}%)"


def format_share(count: int, total: int, digits: int = 2) -> str:
    if total <= 0:
        return "0.00%"
    return f"{count / total * 100:.{digits}f}%"


def generate_markdown_report(stats_by_label: Dict[str, Dict[str, Any]], top_k_md: int = 10) -> str:
    labels = list(stats_by_label.keys())
    baseline = labels[0]

    md: List[str] = []
    md.append("# Result Folder Comparison")
    md.append("")
    md.append("Compared folders:")
    for label in labels:
        md.append(f"- {label}: {stats_by_label[label]['folder']}")

    md.append("")
    md.append("## 1) Basic Scale")
    md.append("")
    md.append("| folder | pkl_files | loaded_files | total_paths | total_nodes | avg_nodes_per_path |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for label in labels:
        s = stats_by_label[label]
        md.append(
            f"| {label} | {s['pkl_files']} | {s['loaded_files']} | {s['total_paths']} | {s['total_nodes']} | {s['avg_nodes_per_path']:.3f} |"
        )

    md.append("")
    md.append("## 2) Node Trigger Frequency")
    md.append("")
    all_nodes = sorted({n for lb in labels for n in stats_by_label[lb]["node_counter"].keys()})
    header = "| node_type | " + " | ".join([f"{lb} count(%)" for lb in labels]) + " |"
    sep = "|---|" + "|".join(["---:"] * len(labels)) + "|"
    md.append(header)
    md.append(sep)
    for node_type in all_nodes:
        row = [node_type]
        for lb in labels:
            s = stats_by_label[lb]
            cnt = s["node_counter"].get(node_type, 0)
            row.append(format_count_pct(cnt, s["total_nodes"]))
        md.append("| " + " | ".join(row) + " |")

    md.append("")
    md.append(f"## 3) Node Order Pattern (Top {top_k_md} Path Sequences)")
    md.append("")
    for lb in labels:
        s = stats_by_label[lb]
        md.append(f"### {lb}")
        md.append("| rank | node_sequence | count | share_in_paths |")
        md.append("|---:|---|---:|---:|")
        for i, (path_str, cnt) in enumerate(s["top_node_paths"][:top_k_md], 1):
            md.append(f"| {i} | {path_str} | {cnt} | {format_share(cnt, s['total_paths'])} |")
        md.append("")

    md.append(f"## 4) Action Path Frequency (Top {top_k_md})")
    md.append("")
    for lb in labels:
        s = stats_by_label[lb]
        md.append(f"### {lb}")
        md.append("| rank | action_path | count | share_in_paths |")
        md.append("|---:|---|---:|---:|")
        for i, (path_str, cnt) in enumerate(s["top_action_paths"][:top_k_md], 1):
            md.append(f"| {i} | {path_str} | {cnt} | {format_share(cnt, s['total_paths'])} |")
        md.append("")

    md.append("## 5) Position-wise Node Distribution")
    md.append("")
    for lb in labels:
        s = stats_by_label[lb]
        md.append(f"### {lb}")
        pos = s["position_node_counter"]
        for step_str in sorted(pos.keys(), key=lambda x: int(x)):
            step_counter = Counter(pos[step_str])
            total_step = sum(step_counter.values())
            top_items = _safe_sort_counter(step_counter)[:5]
            detail = ", ".join(
                [f"{name}: {cnt} ({cnt / total_step * 100:.1f}%)" for name, cnt in top_items]
            ) if total_step > 0 else ""
            md.append(f"- step {step_str}: {detail}")
        md.append("")

    md.append(f"## 6) High-level Delta vs {baseline}")
    md.append("")

    base_stats = stats_by_label[baseline]
    base_total_nodes = max(base_stats["total_nodes"], 1)

    for lb in labels[1:]:
        cur = stats_by_label[lb]
        md.append(f"### {lb} vs {baseline}")

        path_delta = cur["total_paths"] - base_stats["total_paths"]
        avg_delta = cur["avg_nodes_per_path"] - base_stats["avg_nodes_per_path"]

        md.append(
            f"- total_paths: {cur['total_paths']} vs {base_stats['total_paths']} ({path_delta:+d})"
        )
        md.append(
            f"- avg_nodes_per_path: {cur['avg_nodes_per_path']:.3f} vs {base_stats['avg_nodes_per_path']:.3f} ({avg_delta:+.3f})"
        )
        md.append("- node ratio delta (percentage points):")

        cur_total_nodes = max(cur["total_nodes"], 1)
        node_union = sorted(set(base_stats["node_counter"].keys()) | set(cur["node_counter"].keys()))
        for node_type in node_union:
            base_ratio = base_stats["node_counter"].get(node_type, 0) / base_total_nodes * 100
            cur_ratio = cur["node_counter"].get(node_type, 0) / cur_total_nodes * 100
            md.append(f"  - {node_type}: {cur_ratio - base_ratio:+.2f} pp")
        md.append("")

    return "\n".join(md).rstrip() + "\n"


def resolve_target_path(folder_arg: str, results_root: Path, model: str, dataset: str, split: str) -> Path:
    folder_candidate = Path(folder_arg)
    if folder_candidate.exists():
        return folder_candidate.resolve()
    return (results_root / folder_arg / model / dataset / split).resolve()


def make_display_path(abs_path: Path, project_root: Path) -> str:
    try:
        return str(abs_path.relative_to(project_root))
    except ValueError:
        return str(abs_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare node/action path statistics across three result folders."
    )
    parser.add_argument(
        "--folders",
        nargs=3,
        default=["origin1", "llm_guided_fix_1_1", "llm_guided1_3"],
        help="Three folder names under results/, or three explicit folder paths.",
    )
    parser.add_argument("--results-root", default="results", help="Root directory of result folders.")
    parser.add_argument("--model", default="Qwen2.5-Coder-7B-Instruct", help="Model subfolder name.")
    parser.add_argument("--dataset", default="bird", help="Dataset subfolder name.")
    parser.add_argument("--split", default="dev", help="Dataset split subfolder name.")
    parser.add_argument(
        "--labels",
        nargs=3,
        default=None,
        help="Optional labels for the three compared folders. Defaults to folder names.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output file prefix. Defaults to comparison_<label1>_<label2>_<label3>.",
    )
    parser.add_argument("--top-k-md", type=int, default=10, help="Top-K rows to show in Markdown tables.")
    parser.add_argument("--top-k-json", type=int, default=20, help="Top-K paths to store in JSON output.")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    results_root = Path(args.results_root).resolve()

    labels = args.labels if args.labels else [Path(f).name for f in args.folders]

    folder_paths: List[Path] = []
    for folder_arg in args.folders:
        path = resolve_target_path(folder_arg, results_root, args.model, args.dataset, args.split)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Folder not found: {path}")
        folder_paths.append(path)

    stats_by_label: Dict[str, Dict[str, Any]] = {}
    for label, folder_path in zip(labels, folder_paths):
        stats = analyze_folder(folder_path, top_k_json=args.top_k_json)
        stats["folder"] = make_display_path(folder_path, project_root)
        stats_by_label[label] = stats

    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = "comparison_" + "_".join(labels)

    json_path = project_root / f"{output_prefix}.json"
    md_path = project_root / f"{output_prefix}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats_by_label, f, indent=2, ensure_ascii=False)

    md_report = generate_markdown_report(stats_by_label, top_k_md=args.top_k_md)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_report)

    print(f"Generated JSON: {json_path}")
    print(f"Generated Markdown: {md_path}")


if __name__ == "__main__":
    main()
