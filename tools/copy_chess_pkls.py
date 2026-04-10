import argparse
import json
import shutil
from pathlib import Path


def load_question_ids(chess_json_path: Path) -> list[int]:
    with chess_json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    question_ids = []
    for item in data:
        if "question_id" in item:
            question_ids.append(int(item["question_id"]))

    return sorted(set(question_ids))


def find_pkl_candidates(source_root: Path, question_id: int) -> list[Path]:
    target_name = f"{question_id}.pkl"
    return sorted(path for path in source_root.rglob(target_name) if path.is_file())


def copy_matching_pkls(
    chess_json_path: Path,
    source_root: Path,
    destination_root: Path,
    dry_run: bool,
) -> None:
    question_ids = load_question_ids(chess_json_path)
    destination_root.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    copied_question_ids = 0
    missing = []
    multi_match_question_ids = []

    for question_id in question_ids:
        candidates = find_pkl_candidates(source_root, question_id)

        if not candidates:
            missing.append(question_id)
            continue

        if len(candidates) > 1:
            multi_match_question_ids.append(question_id)

        for selected in candidates:
            relative_path = selected.relative_to(source_root)
            dst_path = destination_root / relative_path

            print(f"[COPY] {selected} -> {dst_path}")
            if not dry_run:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(selected, dst_path)
            copied_files += 1

        copied_question_ids += 1

    print("\n===== Summary =====")
    print(f"Question IDs in chess.json: {len(question_ids)}")
    print(f"Copied question_id count: {copied_question_ids}")
    print(f"Copied file count: {copied_files}")
    print(f"Missing: {len(missing)}")
    print(
        "Question IDs with multiple source matches "
        f"(all copied with preserved folder structure): {len(multi_match_question_ids)}"
    )

    if missing:
        print("Missing question_id:")
        print(", ".join(map(str, missing)))

    if multi_match_question_ids:
        print("Question IDs with multiple matches:")
        print(", ".join(map(str, multi_match_question_ids)))


def _resolve_source_root(source_root_arg: Path, workspace_root: Path) -> Path:
    if source_root_arg.exists():
        return source_root_arg.resolve()

    default_results_roots = [
        workspace_root / "oldAlpha" / "b-Alpha-SQL" / "results",
        workspace_root / "oldAlpha" / "b-Alpha-SQL" / "result",
    ]
    for base_root in default_results_roots:
        candidate = base_root / source_root_arg
        if candidate.exists():
            return candidate.resolve()

    return source_root_arg


def _default_chess_json(repo_root: Path) -> Path:
    candidates = [
        repo_root / "data" / "bird" / "dev" / "chess.json",
        repo_root / "data" / "dev" / "chess.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _default_source_root(workspace_root: Path) -> Path:
    candidates = [
        workspace_root / "oldAlpha" / "b-Alpha-SQL" / "results",
        workspace_root / "oldAlpha" / "b-Alpha-SQL" / "result",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _default_destination_root(repo_root: Path) -> Path:
    candidates = [
        repo_root / "results",
        repo_root / "result",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def main() -> None:
    current_file = Path(__file__).resolve()
    alpha_sql_root = current_file.parents[1]
    workspace_root = alpha_sql_root.parent

    parser = argparse.ArgumentParser(
        description="Copy pkl files matching question_id in chess.json."
    )
    parser.add_argument(
        "--chess-json",
        type=Path,
        default=_default_chess_json(alpha_sql_root),
        help="Path to chess.json.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=_default_source_root(workspace_root),
        help="Root directory to search for source pkl files.",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=_default_destination_root(alpha_sql_root),
        help="Destination directory for copied pkl files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be copied.",
    )

    args = parser.parse_args()

    args.source_root = _resolve_source_root(args.source_root, workspace_root)

    if not args.chess_json.exists():
        raise FileNotFoundError(f"chess.json not found: {args.chess_json}")
    if not args.source_root.exists():
        raise FileNotFoundError(f"Source root not found: {args.source_root}")

    copy_matching_pkls(
        chess_json_path=args.chess_json,
        source_root=args.source_root,
        destination_root=args.destination_root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()