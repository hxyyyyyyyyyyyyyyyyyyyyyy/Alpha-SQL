import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


StateKey = Tuple[str, ...]


def normalize_node_name(node_name: str) -> str:
    name = node_name.strip().upper()
    alias_map = {
        "START_NODE": "ROOT",
        "END_NODE": "END",
        "RAPHRASE_QUESTION": "REPHRASE_QUESTION",
        "RAPRASE_QUESTION": "REPHRASE_QUESTION",
    }
    return alias_map.get(name, name)


def is_functional_node(node_name: str) -> bool:
    return node_name not in {"ROOT", "END"}


def state_key_to_str(state_key: StateKey) -> str:
    return "||".join(state_key)


def state_key_from_str(state_key: str) -> StateKey:
    parts = tuple(item for item in state_key.split("||") if item)
    return parts


class TransitionProbabilityPreprocessor:
    def __init__(self, summary_paths_file: str):
        self.summary_paths_file = Path(summary_paths_file)
        if not self.summary_paths_file.exists():
            raise FileNotFoundError(f"summary_paths file not found: {self.summary_paths_file}")

        self.case_sequences: Dict[str, List[List[str]]] = {}

    def load_paths(self) -> Dict[str, List[List[str]]]:
        with self.summary_paths_file.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        case_sequences: Dict[str, List[List[str]]] = {}
        for case_id, paths in raw_data.items():
            if not isinstance(paths, dict):
                continue

            sequences: List[List[str]] = []
            for _, nodes in paths.items():
                if not isinstance(nodes, list):
                    continue
                normalized = [normalize_node_name(node) for node in nodes]
                if normalized and normalized[0] == "ROOT":
                    sequences.append(normalized)

            if sequences:
                case_sequences[str(case_id)] = sequences

        if not case_sequences:
            raise ValueError("No valid paths found in summary_paths file")

        self.case_sequences = case_sequences
        return case_sequences

    def compute_statistics(self) -> Dict[str, Dict]:
        if not self.case_sequences:
            self.load_paths()

        global_state_transition_counts: Dict[StateKey, Counter] = defaultdict(Counter)
        global_node_transition_counts: Dict[str, Counter] = defaultdict(Counter)

        for _, sequences in self.case_sequences.items():
            for seq in sequences:
                for idx in range(len(seq) - 1):
                    current_node = seq[idx]
                    next_node = seq[idx + 1]

                    state_key: StateKey = tuple(seq[: idx + 1])
                    global_state_transition_counts[state_key][next_node] += 1
                    global_node_transition_counts[current_node][next_node] += 1

        return {
            "case_sequences": self.case_sequences,
            "global_state_transition_counts": global_state_transition_counts,
            "global_node_transition_counts": global_node_transition_counts,
        }

    @staticmethod
    def _counter_to_int(counter: Counter) -> Dict[str, int]:
        return {k: int(v) for k, v in counter.items()}

    def build_summary(self) -> Dict:
        stats = self.compute_statistics()

        global_state_transition_matrix = {
            state_key_to_str(state_key): self._counter_to_int(counter)
            for state_key, counter in stats["global_state_transition_counts"].items()
        }

        global_node_transition_matrix = {
            current_node: self._counter_to_int(counter)
            for current_node, counter in stats["global_node_transition_counts"].items()
        }

        return {
            "meta": {
                "summary_paths_file": str(self.summary_paths_file),
                "num_cases": len(stats["case_sequences"]),
                "num_paths": sum(len(paths) for paths in stats["case_sequences"].values()),
            },
            "global_state_transition_matrix": global_state_transition_matrix,
            "global_node_transition_matrix": global_node_transition_matrix,
        }

    def write_summary(self, output_file: str) -> Dict:
        summary = self.build_summary()
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute transition probabilities from summary_paths.json")
    parser.add_argument("--summary-path", type=str, required=True, help="Path to summary_paths.json")
    parser.add_argument("--output", type=str, required=True, help="Output summary json path")
    return parser.parse_args()


def main():
    args = _parse_args()
    preprocessor = TransitionProbabilityPreprocessor(summary_paths_file=args.summary_path)
    preprocessor.write_summary(args.output)
    print(f"Transition probabilities summary saved to {args.output}")


if __name__ == "__main__":
    main()
