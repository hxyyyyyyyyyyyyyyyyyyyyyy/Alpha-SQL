#!/usr/bin/env python3
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
BALL_PATH = ROOT / "results" / "ball.json"

# (name, old_dir, new_dir, output_dir)
JOBS = [
    (
        "llm_guide24_eps0.2",
        ROOT / "results_raw" / "old" / "llm_guide24_eps0.2" / "bird" / "dev",
        ROOT / "results_raw" / "new" / "llm_guide24_eps0.2" / "Qwen2.5-Coder-7B-Instruct" / "bird" / "dev",
        ROOT / "results" / "llm_guide24_eps0.2" / "bird" / "dev",
    ),
    (
        "llm_guide24_eps0.5",
        ROOT / "results_raw" / "old" / "llm_guide24_eps0.5" / "bird" / "dev",
        ROOT / "results_raw" / "new" / "llm_guide24_eps0.5" / "Qwen2.5-Coder-7B-Instruct" / "bird" / "dev",
        ROOT / "results" / "llm_guide24_eps0.5" / "bird" / "dev",
    ),
    (
        "llm_with_nodescore_eps0.2",
        ROOT / "results_raw" / "old" / "llm_with_nodescore_eps0.2" / "bird" / "dev",
        ROOT / "results_raw" / "new" / "llm_with_nodescore_eps0.2" / "Qwen2.5-Coder-7B-Instruct" / "bird" / "dev",
        ROOT / "results" / "llm_with_nodescore_eps0.2" / "bird" / "dev",
    ),
    (
        "llm_with_nodescore_eps0.5",
        ROOT / "results_raw" / "old" / "llm_with_nodescore_eps0.5" / "bird" / "dev",
        ROOT / "results_raw" / "new" / "llm_with_nodescore_eps0.5" / "Qwen2.5-Coder-7B-Instruct" / "bird" / "dev",
        ROOT / "results" / "llm_with_nodescore_eps0.5" / "bird" / "dev",
    ),
    (
        "origin_qwen2.5",
        ROOT / "results_raw" / "old" / "origin" / "Qwen2.5-Coder-7B-Instruct" / "bird" / "dev",
        ROOT / "results_raw" / "new" / "origin24" / "Qwen2.5-Coder-7B-Instruct" / "bird" / "dev",
        ROOT / "results" / "origin" / "Qwen2.5-Coder-7B-Instruct" / "bird" / "dev",
    ),
]


def load_ball_ids(ball_path: Path) -> set[int]:
    with ball_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(item["question_id"]) for item in data if "question_id" in item}


def parse_id_from_pkl(p: Path) -> int:
    # filenames are like 1020.pkl
    return int(p.stem)


def choose_source(qid: int, old_file: Path, new_file: Path, use_new_ids: set[int]) -> tuple[Path | None, str]:
    preferred_new = qid in use_new_ids

    if preferred_new:
        if new_file.exists():
            return new_file, "new"
        if old_file.exists():
            return old_file, "old_fallback"
        return None, "missing"

    if old_file.exists():
        return old_file, "old"
    if new_file.exists():
        return new_file, "new_fallback"
    return None, "missing"


def run_job(name: str, old_dir: Path, new_dir: Path, out_dir: Path, use_new_ids: set[int]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    old_files = {p.name: p for p in old_dir.glob("*.pkl")} if old_dir.exists() else {}
    new_files = {p.name: p for p in new_dir.glob("*.pkl")} if new_dir.exists() else {}

    all_names = sorted(set(old_files) | set(new_files), key=lambda x: int(Path(x).stem))

    stats = {
        "job": name,
        "total_candidates": len(all_names),
        "copied": 0,
        "from_old": 0,
        "from_new": 0,
        "old_fallback": 0,
        "new_fallback": 0,
        "missing": 0,
    }

    for fname in all_names:
        qid = parse_id_from_pkl(Path(fname))
        old_file = old_files.get(fname, old_dir / fname)
        new_file = new_files.get(fname, new_dir / fname)

        src, tag = choose_source(qid, old_file, new_file, use_new_ids)
        if src is None:
            stats["missing"] += 1
            continue

        dst = out_dir / fname
        shutil.copy2(src, dst)
        stats["copied"] += 1

        if tag == "old":
            stats["from_old"] += 1
        elif tag == "new":
            stats["from_new"] += 1
        elif tag == "old_fallback":
            stats["old_fallback"] += 1
        elif tag == "new_fallback":
            stats["new_fallback"] += 1

    return stats


def main() -> None:
    use_new_ids = load_ball_ids(BALL_PATH)

    all_stats = []
    for job in JOBS:
        all_stats.append(run_job(*job, use_new_ids))

    print("Copy done. Summary:")
    for s in all_stats:
        print(
            f"- {s['job']}: candidates={s['total_candidates']}, copied={s['copied']}, "
            f"new={s['from_new']}, old={s['from_old']}, "
            f"old_fallback={s['old_fallback']}, new_fallback={s['new_fallback']}, missing={s['missing']}"
        )


if __name__ == "__main__":
    main()
