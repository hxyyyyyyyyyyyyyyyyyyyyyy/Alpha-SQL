#!/usr/bin/env python3

import argparse
import pickle
import shutil
import sys
from pathlib import Path
from typing import Optional


SUBSET_SIZES = (4, 8, 12, 16, 20)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def resolve_results_root(project_root: Path, results_root_arg: Optional[str]) -> Path:
    if results_root_arg:
        candidate = Path(results_root_arg)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return candidate

    preferred = project_root / "results" / "llm_guided24"
    fallback = project_root / "results" / "llm_guide24"

    if preferred.exists():
        return preferred
    return fallback


def discover_source_model_dirs(results_root: Path):
    model_dirs = []
    if not results_root.exists():
        return model_dirs

    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        if "_subset_" in child.name:
            continue
        dev_dir = child / "bird" / "dev"
        if dev_dir.exists() and any(dev_dir.glob("*.pkl")):
            model_dirs.append(child)
    return model_dirs


def load_rollouts(pkl_path: Path):
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{pkl_path} does not contain a list of rollouts")
    return data


def is_subset_already_complete(subset_dev_dir: Path, source_pkl_files, subset_size: int) -> bool:
    if not subset_dev_dir.exists():
        return False

    subset_pkls = sorted(subset_dev_dir.glob("*.pkl"), key=sort_key)
    if len(subset_pkls) != len(source_pkl_files):
        return False

    source_names = [p.name for p in source_pkl_files]
    subset_names = [p.name for p in subset_pkls]
    if source_names != subset_names:
        return False

    # Assume subset is completed when data point files are aligned.
    # We avoid loading every pkl here to keep the check lightweight.
    print(
        f"[SKIP] Subset already completed: {subset_dev_dir} "
        f"({len(subset_pkls)} data points, target {subset_size} rollouts each)"
    )
    return True


def extract_subsets_for_model(model_dir: Path, strict: bool) -> int:
    dev_dir = model_dir / "bird" / "dev"
    pkl_files = sorted(dev_dir.glob("*.pkl"), key=sort_key)

    if not pkl_files:
        print(f"[WARN] No pkl files found in: {dev_dir}")
        return 0

    parent_root = model_dir.parent
    model_name = model_dir.name
    success_count = 0

    for size in SUBSET_SIZES:
        subset_name = f"{model_name}_subset_{size}"
        subset_dev_dir = parent_root / subset_name / "bird" / "dev"

        if is_subset_already_complete(subset_dev_dir, pkl_files, size):
            success_count += 1
            continue

        subset_dev_dir.mkdir(parents=True, exist_ok=True)

        # Avoid stale files from previous runs.
        for old_file in subset_dev_dir.glob("*.pkl"):
            old_file.unlink()

        subset_failed = False
        written_count = 0

        for pkl_file in pkl_files:
            try:
                rollouts = load_rollouts(pkl_file)
            except Exception as e:
                print(f"[ERROR] Failed to read {pkl_file}: {e}")
                subset_failed = True
                if strict:
                    return 1
                break

            if len(rollouts) < size:
                print(
                    f"[ERROR] Not enough rollouts in {pkl_file.name} for {subset_name}: "
                    f"need {size}, found {len(rollouts)}"
                )
                subset_failed = True
                if strict:
                    return 1
                break

            out_path = subset_dev_dir / pkl_file.name
            with out_path.open("wb") as f:
                pickle.dump(rollouts[:size], f)
            written_count += 1

        if subset_failed:
            for generated in subset_dev_dir.glob("*.pkl"):
                generated.unlink()
            continue

        config_file = dev_dir / "config.json"
        if config_file.exists():
            shutil.copy2(config_file, subset_dev_dir / config_file.name)

        print(
            f"[OK] Generated subset: {subset_dev_dir} "
            f"({written_count} data points, {size} rollouts each)"
        )
        success_count += 1

    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract rollout subsets (4,8,12,16,20) for each data point."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Root folder containing model result dirs (default: results/llm_guided24, fallback results/llm_guide24)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code when a subset size cannot be generated.",
    )
    args = parser.parse_args()

    results_root = resolve_results_root(PROJECT_ROOT, args.results_root)

    if not results_root.exists():
        print(f"[ERROR] Results root not found: {results_root}")
        return 1

    model_dirs = discover_source_model_dirs(results_root)
    if not model_dirs:
        print(f"[ERROR] No source model directories found under: {results_root}")
        print("Expected structure: <model_name>/bird/dev/*.pkl")
        return 1

    print(f"[INFO] Using results root: {results_root}")
    print(f"[INFO] Found {len(model_dirs)} source model folder(s)")

    total_success = 0
    for model_dir in model_dirs:
        print(f"[INFO] Processing model: {model_dir.name}")
        result = extract_subsets_for_model(model_dir, strict=args.strict)
        if result == 1 and args.strict:
            return 1
        total_success += result

    print(f"[INFO] Done. Generated {total_success} subset folder(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
