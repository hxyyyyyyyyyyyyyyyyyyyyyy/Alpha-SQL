#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


def load_question_ids(ball_json_path: Path) -> Set[int]:
    payload = json.loads(ball_json_path.read_text(encoding="utf-8"))
    qids: Set[int] = set()
    for item in payload:
        if isinstance(item, dict) and "question_id" in item:
            try:
                qids.add(int(item["question_id"]))
            except Exception:
                continue
    return qids


def is_under_excluded_dir(path: Path, excluded_dir_name: str) -> bool:
    return excluded_dir_name in path.parts


def collect_target_files(
    results_root: Path,
    question_ids: Set[int],
    excluded_dir_name: str,
    suffixes: Set[str],
) -> List[Path]:
    targets: List[Path] = []
    for fp in results_root.rglob("*"):
        if not fp.is_file():
            continue
        if is_under_excluded_dir(fp, excluded_dir_name):
            continue
        if fp.suffix.lower() not in suffixes:
            continue
        stem = fp.stem
        if not stem.isdigit():
            continue
        if int(stem) in question_ids:
            targets.append(fp)
    return sorted(targets)


def group_by_top_level(paths: List[Path], results_root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for p in paths:
        rel = p.relative_to(results_root)
        top = rel.parts[0] if len(rel.parts) > 0 else "."
        counts[top] += 1
    return dict(sorted(counts.items(), key=lambda x: x[0]))


def parse_suffixes(s: str) -> Set[str]:
    vals = set()
    for raw in s.split(","):
        x = raw.strip().lower()
        if not x:
            continue
        if not x.startswith("."):
            x = f".{x}"
        vals.add(x)
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete datapoint files whose numeric stem matches question_ids in ball.json, excluding a specific folder name."
    )
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--ball-json", type=str, default="results/ball.json")
    parser.add_argument("--exclude-dir-name", type=str, default="origin")
    parser.add_argument("--suffixes", type=str, default=".pkl", help="Comma-separated suffix list, e.g. .pkl,.json")
    parser.add_argument("--apply", action="store_true", help="Actually delete files. Without this flag, only dry-run.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    ball_json_path = Path(args.ball_json)
    suffixes = parse_suffixes(args.suffixes)

    if not results_root.exists():
        raise FileNotFoundError(f"results root not found: {results_root}")
    if not ball_json_path.exists():
        raise FileNotFoundError(f"ball json not found: {ball_json_path}")

    question_ids = load_question_ids(ball_json_path)
    targets = collect_target_files(
        results_root=results_root,
        question_ids=question_ids,
        excluded_dir_name=args.exclude_dir_name,
        suffixes=suffixes,
    )

    print(f"[info] question_ids in ball.json: {len(question_ids)}")
    print(f"[info] excluded directory name: {args.exclude_dir_name}")
    print(f"[info] suffixes: {sorted(suffixes)}")
    print(f"[info] matched files: {len(targets)}")

    by_top = group_by_top_level(targets, results_root)
    if by_top:
        print("[info] matched count by results/<top-level>:")
        for top, cnt in by_top.items():
            print(f"  - {top}: {cnt}")

    if args.verbose:
        for fp in targets:
            print(fp.as_posix())

    if not args.apply:
        print("[dry-run] no file deleted. add --apply to execute deletion.")
        return

    deleted = 0
    for fp in targets:
        try:
            fp.unlink()
            deleted += 1
        except FileNotFoundError:
            continue
    print(f"[done] deleted files: {deleted}")


if __name__ == "__main__":
    main()
