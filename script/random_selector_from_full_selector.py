import argparse
import pickle
import random
import sys
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sample_results(
    source_dir: Path,
    destination_dir: Path,
    sample_size: int,
    expected_size: int,
    strict_expected_size: bool,
    seed: int,
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    rng = random.Random(seed)
    destination_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(source_dir.glob("*.pkl"))
    if not pkl_files:
        raise ValueError(f"No pkl files found in: {source_dir}")

    processed = 0
    skipped = 0

    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as file:
            results = pickle.load(file)

        if not isinstance(results, list):
            raise TypeError(f"Expected list in {pkl_path}, got {type(results)}")

        if strict_expected_size and len(results) != expected_size:
            raise ValueError(
                f"{pkl_path.name} has {len(results)} results, expected exactly {expected_size}"
            )

        if len(results) < sample_size:
            print(
                f"[SKIP] {pkl_path.name}: result count {len(results)} < sample_size {sample_size}"
            )
            skipped += 1
            continue

        sampled = rng.sample(results, sample_size)

        save_path = destination_dir / pkl_path.name
        with open(save_path, "wb") as file:
            pickle.dump(sampled, file)

        processed += 1

    print("===== Summary =====")
    print(f"Source dir: {source_dir}")
    print(f"Destination dir: {destination_dir}")
    print(f"Total pkl files: {len(pkl_files)}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Sample size per file: {sample_size}")
    print(f"Seed: {seed}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read full_selector pkl results, randomly sample 4 paths from each file, "
            "and save into random_selector4 directory."
        )
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("results/full_selector/Qwen2.5-Coder-7B-Instruct/bird/dev"),
        help="Directory containing full_selector per-question pkl files.",
    )
    parser.add_argument(
        "--destination_dir",
        type=Path,
        default=Path("results/random_selector4/Qwen2.5-Coder-7B-Instruct/bird/dev"),
        help="Directory to save sampled per-question pkl files.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=4,
        help="Number of random results sampled from each pkl file.",
    )
    parser.add_argument(
        "--expected_size",
        type=int,
        default=64,
        help="Expected full_selector result count in each pkl file.",
    )
    parser.add_argument(
        "--strict_expected_size",
        action="store_true",
        help="If set, require each pkl file length equals expected_size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    sample_results(
        source_dir=args.source_dir,
        destination_dir=args.destination_dir,
        sample_size=args.sample_size,
        expected_size=args.expected_size,
        strict_expected_size=args.strict_expected_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()