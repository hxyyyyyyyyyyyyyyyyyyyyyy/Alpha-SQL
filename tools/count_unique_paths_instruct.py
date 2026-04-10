import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable, List


def _node_type_to_str(node: object) -> str:
    node_type = getattr(node, "node_type", None)
    if node_type is None:
        return type(node).__name__

    value = getattr(node_type, "value", None)
    if value is not None:
        return str(value)
    return str(node_type)


def _path_signature(path_nodes: object) -> str:
    if isinstance(path_nodes, (list, tuple)):
        if not path_nodes:
            return "EMPTY_PATH"
        return "->".join(_node_type_to_str(node) for node in path_nodes)
    return f"NON_LIST_PATH:{type(path_nodes).__name__}"


def _count_unique_paths_in_pkl(pkl_path: Path) -> dict:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        return {
            "unique_path_count": 0,
            "total_path_count": 0,
            "error": f"Unexpected pickle root type: {type(data).__name__}",
        }

    signatures = {_path_signature(path_nodes) for path_nodes in data}
    return {
        "unique_path_count": len(signatures),
        "total_path_count": len(data),
        "error": None,
    }


def _iter_instruct_dirs(root_dir: Path) -> Iterable[Path]:
    for child in sorted(root_dir.iterdir()):
        if child.is_dir() and child.name.endswith("Instruct"):
            yield child


def analyze_one_instruct_dir(instruct_dir: Path, output_filename: str) -> Path:
    pkl_files: List[Path] = sorted(instruct_dir.rglob("*.pkl"))
    per_pkl = []
    distribution_counter: Counter = Counter()

    for pkl_path in pkl_files:
        result = _count_unique_paths_in_pkl(pkl_path)
        unique_path_count = result["unique_path_count"]

        if result["error"] is None:
            distribution_counter[unique_path_count] += 1

        per_pkl.append(
            {
                "file": str(pkl_path.relative_to(instruct_dir)),
                "unique_path_count": unique_path_count,
                "total_path_count": result["total_path_count"],
                "error": result["error"],
            }
        )

    output = {
        "instruct_dir": str(instruct_dir),
        "num_pkl_files": len(pkl_files),
        "path_count_distribution": {
            str(path_count): data_points
            for path_count, data_points in sorted(distribution_counter.items(), key=lambda x: x[0])
        },
        "per_pkl": per_pkl,
    }

    output_path = instruct_dir / output_filename
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count unique path templates in each pkl under *Instruct folders, "
            "and aggregate distribution of unique-path-count -> number of data points."
        )
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="results/llm_guided6",
        help="Root directory that contains *Instruct subdirectories.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="unique_path_count_stats.json",
        help="Output JSON filename to save inside each *Instruct folder.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    if not root_dir.exists() or not root_dir.is_dir():
        raise ValueError(f"Invalid root_dir: {root_dir}")

    instruct_dirs = list(_iter_instruct_dirs(root_dir))
    if not instruct_dirs:
        raise ValueError(f"No *Instruct directories found under: {root_dir}")

    for instruct_dir in instruct_dirs:
        output_path = analyze_one_instruct_dir(instruct_dir, args.output_filename)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
