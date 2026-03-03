import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from alphasql.algorithm.path_selector.transition_preprocessor import (
    TransitionProbabilityPreprocessor,
    is_functional_node,
    state_key_from_str,
)


StateKey = Tuple[str, ...]


class RandomPathSelector:
    """
    基于 summary_paths.json / 预处理概率文件 的概率随机路径选择器。

    约束：一次采样路径内，功能节点不可重复。
    """

    def __init__(
        self,
        summary_paths_file: Optional[str] = None,
        probability_summary_file: Optional[str] = None,
        random_seed: Optional[int] = None,
    ):
        if summary_paths_file is None and probability_summary_file is None:
            raise ValueError("Either summary_paths_file or probability_summary_file must be provided")

        self.random = random.Random(random_seed)

        self.global_state_transition_counts: Dict[StateKey, Counter] = {}
        self.global_node_transition_counts: Dict[str, Counter] = {}

        if probability_summary_file:
            self._load_from_probability_summary(probability_summary_file)
        else:
            self._load_from_summary_paths(summary_paths_file)

    def _load_from_summary_paths(self, summary_paths_file: str):
        preprocessor = TransitionProbabilityPreprocessor(summary_paths_file=summary_paths_file)
        stats = preprocessor.compute_statistics()
        self.global_state_transition_counts = stats["global_state_transition_counts"]
        self.global_node_transition_counts = stats["global_node_transition_counts"]

    def _load_from_probability_summary(self, probability_summary_file: str):
        path = Path(probability_summary_file)
        if not path.exists():
            raise FileNotFoundError(f"probability summary file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        state_matrix = summary.get("global_state_transition_matrix")
        if state_matrix is None:
            state_matrix = summary.get("global_state_transition_probabilities", {})
        node_matrix = summary.get("global_node_transition_matrix", {})

        self.global_state_transition_counts = {
            state_key_from_str(state_key): Counter({k: float(v) for k, v in counts.items()})
            for state_key, counts in state_matrix.items()
        }
        self.global_node_transition_counts = {
            current_node: Counter({k: float(v) for k, v in counts.items()})
            for current_node, counts in node_matrix.items()
        }

    def _weighted_choice(self, counter: Counter) -> Optional[str]:
        if not counter:
            return None
        total = sum(counter.values())
        if total <= 0:
            return None

        threshold = self.random.uniform(0, total)
        cumulative = 0.0
        for name, weight in counter.items():
            cumulative += weight
            if cumulative >= threshold:
                return name
        return next(iter(counter.keys()))

    def _get_candidate_distribution(self, path_prefix: List[str], used_function_nodes: Set[str]) -> Counter:
        state_key: StateKey = tuple(path_prefix)
        current_node = path_prefix[-1]
        merged = Counter()
        merged.update(self.global_state_transition_counts.get(state_key, Counter()))
        if not merged:
            merged.update(self.global_node_transition_counts.get(current_node, Counter()))

        filtered = Counter()
        for next_node, weight in merged.items():
            if is_functional_node(next_node) and next_node in used_function_nodes:
                continue
            filtered[next_node] = weight
        return filtered

    def sample_path(self, case_id: Optional[str] = None, max_steps: int = 16) -> List[str]:
        path = ["ROOT"]
        used_function_nodes: Set[str] = set()

        for _ in range(max_steps):
            candidates = self._get_candidate_distribution(path, used_function_nodes)
            next_node = self._weighted_choice(candidates)
            if next_node is None:
                break

            path.append(next_node)

            if is_functional_node(next_node):
                used_function_nodes.add(next_node)

            if next_node == "END":
                return path

        if path[-1] != "END":
            path.append("END")
        return path

    def sample_paths(self, num_samples: int, case_id: Optional[str] = None, max_steps: int = 16) -> List[List[str]]:
        return [self.sample_path(case_id=case_id, max_steps=max_steps) for _ in range(num_samples)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random path selector")
    parser.add_argument("--summary-path", type=str, default=None, help="Path to summary_paths.json")
    parser.add_argument("--probability-summary", type=str, default=None, help="Path to transition probabilities summary json")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    selector = RandomPathSelector(
        summary_paths_file=args.summary_path,
        probability_summary_file=args.probability_summary,
        random_seed=args.seed,
    )
    samples = selector.sample_paths(num_samples=args.num_samples, case_id=None, max_steps=args.max_steps)
    print(json.dumps(samples, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
