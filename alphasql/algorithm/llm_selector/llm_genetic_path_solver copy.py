from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import copy
import json
import pickle
import random

from alphasql.algorithm.llm_selector.llm_action_selector import LLMActionSelector
from alphasql.algorithm.mcts.mcts_action import (
    MCTSNodeType,
    EndAction,
    NODE_TYPE_TO_VALID_ACTIONS,
)
from alphasql.algorithm.mcts.mcts_node import MCTSNode, get_valid_action_space_for_node
from alphasql.database.sql_execution import cached_execute_sql_with_timeout, SQLExecutionResultType
from alphasql.database.utils import build_table_ddl_statement
from alphasql.runner.task import Task


BASE_ACTION_POOL = [
    "RaphraseQuestionAction",
    "SchemaSelectionAction",
    "IdentifyColumnValuesAction",
    "IdentifyColumnFunctionsAction",
    "SQLGenerationAction",
]

ACTION_NAME_TO_NEXT_NODE_TYPE = {
    "RaphraseQuestionAction": MCTSNodeType.REPHRASE_QUESTION,
    "SchemaSelectionAction": MCTSNodeType.SCHEMA_SELECTION,
    "IdentifyColumnValuesAction": MCTSNodeType.IDENTIFY_COLUMN_VALUES,
    "IdentifyColumnFunctionsAction": MCTSNodeType.IDENTIFY_COLUMN_FUNCTIONS,
    "SQLGenerationAction": MCTSNodeType.SQL_GENERATION,
    "SQLRevisionAction": MCTSNodeType.SQL_REVISION,
    "EndAction": MCTSNodeType.END,
}


class LLMGeneticPathSolver:
    def __init__(
        self,
        db_root_dir: str,
        task: Task,
        max_depth: int,
        max_path_steps: int,
        save_root_dir: str,
        llm_kwargs: Dict[str, Any],
        llm_seed_num_paths: int,
        llm_seed_num_retry: int,
        target_num_paths: int,
        ga_population_size: int,
        ga_max_generations: int,
        ga_crossover_rate: float,
        ga_mutation_rate: float,
        ga_tournament_size: int,
        random_seed: Optional[int] = 42,
    ):
        self.db_root_dir = db_root_dir
        self.task = task
        self.max_depth = max_depth
        self.max_path_steps = max_path_steps
        self.save_root_dir = save_root_dir
        self.llm_kwargs = llm_kwargs

        self.llm_seed_num_paths = llm_seed_num_paths
        self.llm_seed_num_retry = max(1, llm_seed_num_retry)
        self.target_num_paths = target_num_paths

        self.ga_population_size = ga_population_size
        self.ga_max_generations = ga_max_generations
        self.ga_crossover_rate = ga_crossover_rate
        self.ga_mutation_rate = ga_mutation_rate
        self.ga_tournament_size = max(1, ga_tournament_size)

        self.rng = random.Random(random_seed)
        self.action_selector = LLMActionSelector(llm_kwargs)

        self.generated_records: List[Dict[str, Any]] = []

    def _build_root_node(self) -> MCTSNode:
        if not self.task.table_schema_dict:
            raise ValueError(f"Task {self.task.question_id} has empty table_schema_dict")

        schema_context = "\n".join(
            [
                build_table_ddl_statement(
                    self.task.table_schema_dict[table_name].to_dict(),
                    add_value_description=True,
                    add_column_description=True,
                    add_value_examples=True,
                    add_expanded_column_name=True,
                )
                for table_name in self.task.table_schema_dict
            ]
        )

        root_node = MCTSNode(
            MCTSNodeType.ROOT,
            parent_node=None,
            parent_action=None,
            depth=0,
            db_id=self.task.db_id,
            db_root_dir=self.db_root_dir,
            original_question=self.task.question,
            hint=self.task.evidence,
            schema_context=schema_context,
            table_schema_dict=self.task.table_schema_dict,
        )
        root_node.path_nodes = [root_node]
        return root_node

    def _generate_one_llm_path(self) -> Optional[List[MCTSNode]]:
        current = self._build_root_node()
        path = [current]

        for _ in range(min(self.max_path_steps, self.max_depth)):
            if current.is_terminal():
                break
            valid_action_space = get_valid_action_space_for_node(current)
            if not valid_action_space:
                break

            selected_action = self.action_selector.select_action(current, valid_action_space)
            children_nodes = selected_action.create_children_nodes(current, self.llm_kwargs)
            if not children_nodes:
                break

            current = children_nodes[0]
            current.children = []
            path.append(current)

            if current.is_terminal():
                break

        if not current.is_terminal() and current.node_type in [MCTSNodeType.SQL_GENERATION, MCTSNodeType.SQL_REVISION]:
            end_nodes = EndAction().create_children_nodes(current, self.llm_kwargs)
            if end_nodes:
                current = end_nodes[0]
                path.append(current)

        if path and path[-1].is_terminal():
            return path
        return None

    def _extract_action_sequence(self, path: List[MCTSNode]) -> List[str]:
        return [
            node.parent_action.__class__.__name__
            for node in path
            if node.parent_action is not None
        ]

    def _execution_summary(self, final_sql: Optional[str]) -> Dict[str, Any]:
        if not final_sql:
            return {
                "execution_success": False,
                "result_type": "missing_sql",
                "result_row_count": 0,
                "error_message": "No final SQL generated",
            }

        db_path = str(Path(self.db_root_dir) / self.task.db_id / f"{self.task.db_id}.sqlite")
        execution_result = cached_execute_sql_with_timeout(db_path, final_sql)
        is_success = execution_result.result_type == SQLExecutionResultType.SUCCESS

        return {
            "execution_success": is_success,
            "result_type": execution_result.result_type.value,
            "result_row_count": 0 if execution_result.result is None else len(execution_result.result),
            "error_message": execution_result.error_message,
        }

    def _evaluate_path(self, path: Optional[List[MCTSNode]], source: str, generation: int) -> Dict[str, Any]:
        if not path:
            record = {
                "source": source,
                "generation": generation,
                "action_sequence": [],
                "final_sql": None,
                "execution_success": False,
                "result_type": "invalid_path",
                "result_row_count": 0,
                "error_message": "Path replay failed or path is not terminal",
                "fitness": 0.0,
            }
            self.generated_records.append(record)
            return record

        action_sequence = self._extract_action_sequence(path)
        final_sql = path[-1].final_sql_query
        execution_info = self._execution_summary(final_sql)

        fitness = 0.0
        if execution_info["execution_success"]:
            fitness += 1.0
        if execution_info["result_row_count"] > 0:
            fitness += 0.2
        fitness += 0.02 * len(action_sequence)

        record = {
            "source": source,
            "generation": generation,
            "action_sequence": action_sequence,
            "final_sql": final_sql,
            "execution_success": execution_info["execution_success"],
            "result_type": execution_info["result_type"],
            "result_row_count": execution_info["result_row_count"],
            "error_message": execution_info["error_message"],
            "fitness": fitness,
        }
        self.generated_records.append(record)
        return record

    def _replay_from_action_sequence(self, action_sequence: List[str]) -> Optional[List[MCTSNode]]:
        current = self._build_root_node()
        path = [current]

        for action_name in action_sequence:
            if len(path) - 1 >= min(self.max_path_steps, self.max_depth):
                break
            if current.is_terminal():
                break

            valid_action_space = get_valid_action_space_for_node(current)
            target_action = None
            for action in valid_action_space:
                if action.__class__.__name__ == action_name:
                    target_action = action
                    break
            if target_action is None:
                break

            children_nodes = target_action.create_children_nodes(current, self.llm_kwargs)
            if not children_nodes:
                break

            current = children_nodes[0]
            current.children = []
            path.append(current)

        if not current.is_terminal() and current.node_type in [MCTSNodeType.SQL_GENERATION, MCTSNodeType.SQL_REVISION]:
            end_nodes = EndAction().create_children_nodes(current, self.llm_kwargs)
            if end_nodes:
                current = end_nodes[0]
                path.append(current)

        if path[-1].is_terminal():
            return path
        return None

    def _path_signature(self, record: Dict[str, Any]) -> Tuple[Any, Any]:
        return (tuple(record["action_sequence"]), record["final_sql"])

    def _normalize_sequence(self, action_sequence: List[str]) -> List[str]:
        proposed_actions = [a for a in action_sequence if a in ACTION_NAME_TO_NEXT_NODE_TYPE and a != "EndAction"]
        max_steps = max(1, min(self.max_path_steps, self.max_depth))

        current_node_type = MCTSNodeType.ROOT
        used_action_names = set()
        normalized_actions: List[str] = []

        for action_name in proposed_actions:
            if len(normalized_actions) >= max_steps:
                break

            valid_action_names = [
                action_class.__name__
                for action_class in NODE_TYPE_TO_VALID_ACTIONS[current_node_type]
                if action_class.__name__ not in used_action_names
            ]
            if action_name not in valid_action_names:
                continue

            normalized_actions.append(action_name)
            used_action_names.add(action_name)
            current_node_type = ACTION_NAME_TO_NEXT_NODE_TYPE[action_name]

            if current_node_type == MCTSNodeType.END:
                return normalized_actions

        while len(normalized_actions) < max_steps and current_node_type != MCTSNodeType.END:
            valid_action_names = [
                action_class.__name__
                for action_class in NODE_TYPE_TO_VALID_ACTIONS[current_node_type]
                if action_class.__name__ not in used_action_names
            ]
            if not valid_action_names:
                break

            selected_action_name = None
            for candidate in ["EndAction", "SQLGenerationAction", "SQLRevisionAction"]:
                if candidate in valid_action_names:
                    selected_action_name = candidate
                    break
            if selected_action_name is None:
                selected_action_name = valid_action_names[0]

            normalized_actions.append(selected_action_name)
            used_action_names.add(selected_action_name)
            current_node_type = ACTION_NAME_TO_NEXT_NODE_TYPE[selected_action_name]

        if normalized_actions and normalized_actions[-1] == "EndAction":
            return normalized_actions

        if current_node_type in [MCTSNodeType.SQL_GENERATION, MCTSNodeType.SQL_REVISION] and "EndAction" not in used_action_names:
            normalized_actions.append("EndAction")

        return normalized_actions

    def _select_parent(self, population: List[Tuple[List[MCTSNode], Dict[str, Any]]]) -> Tuple[List[MCTSNode], Dict[str, Any]]:
        tournament_size = min(self.ga_tournament_size, len(population))
        contestants = self.rng.sample(population, tournament_size)
        contestants = sorted(contestants, key=lambda item: item[1]["fitness"], reverse=True)
        return contestants[0]

    def _crossover(self, seq_a: List[str], seq_b: List[str]) -> List[str]:
        core_a = [a for a in seq_a if a != "EndAction"]
        core_b = [a for a in seq_b if a != "EndAction"]
        if not core_a:
            return self._normalize_sequence(seq_b)
        if not core_b:
            return self._normalize_sequence(seq_a)

        cut = self.rng.randint(1, len(core_a))
        prefix = core_a[:cut]
        child_core = prefix + [action for action in core_b if action not in prefix]
        return self._normalize_sequence(child_core + ["EndAction"])

    def _mutate(self, action_sequence: List[str]) -> List[str]:
        core = [a for a in action_sequence if a != "EndAction"]
        if not core:
            core = ["SQLGenerationAction"]

        mutation_type = self.rng.choice(["insert", "remove", "swap"])

        if mutation_type == "insert":
            candidates = [a for a in BASE_ACTION_POOL if a not in core]
            if candidates:
                insert_action = self.rng.choice(candidates)
                insert_pos = self.rng.randint(0, len(core))
                core = core[:insert_pos] + [insert_action] + core[insert_pos:]
        elif mutation_type == "remove":
            removable = [idx for idx, action in enumerate(core) if action != "SQLGenerationAction"]
            if removable:
                remove_idx = self.rng.choice(removable)
                core.pop(remove_idx)
        else:
            if len(core) >= 2:
                i, j = self.rng.sample(range(len(core)), 2)
                core[i], core[j] = core[j], core[i]

        return self._normalize_sequence(core + ["EndAction"])

    def _build_seed_population(self) -> List[Tuple[List[MCTSNode], Dict[str, Any]]]:
        population: List[Tuple[List[MCTSNode], Dict[str, Any]]] = []
        max_attempts = max(self.llm_seed_num_paths * 8, self.llm_seed_num_paths + self.llm_seed_num_retry)
        attempts = 0
        consecutive_retry_count = 0
        seen_signatures = set()

        while (
            len(population) < self.llm_seed_num_paths
            and attempts < max_attempts
            and consecutive_retry_count < self.llm_seed_num_retry
        ):
            attempts += 1
            path = self._generate_one_llm_path()
            record = self._evaluate_path(path, source="llm_seed", generation=0)
            if not record["execution_success"]:
                consecutive_retry_count += 1
                continue

            signature = self._path_signature(record)
            if signature in seen_signatures:
                consecutive_retry_count += 1
            else:
                seen_signatures.add(signature)
                consecutive_retry_count = 0

            population.append((path, record))

        if len(population) < self.llm_seed_num_paths and consecutive_retry_count >= self.llm_seed_num_retry:
            print(
                f"Question ID: {self.task.question_id}, seed generation early stop: "
                f"consecutive retries reached llm_seed_num_retry={self.llm_seed_num_retry}"
            )

        return population

    def solve(self):
        population = self._build_seed_population()
        final_success_paths = [path for path, _ in population]

        generation = 0
        while (
            len(final_success_paths) < self.target_num_paths
            and generation < self.ga_max_generations
            and population
        ):
            generation += 1
            offspring: List[Tuple[List[MCTSNode], Dict[str, Any]]] = []

            while len(offspring) < self.ga_population_size:
                parent_a_path, parent_a_record = self._select_parent(population)
                parent_b_path, parent_b_record = self._select_parent(population)

                seq_a = parent_a_record["action_sequence"]
                seq_b = parent_b_record["action_sequence"]

                child_seq = copy.deepcopy(seq_a)
                if self.rng.random() < self.ga_crossover_rate:
                    child_seq = self._crossover(seq_a, seq_b)
                if self.rng.random() < self.ga_mutation_rate:
                    child_seq = self._mutate(child_seq)
                child_seq = self._normalize_sequence(child_seq)

                child_path = self._replay_from_action_sequence(child_seq)
                child_record = self._evaluate_path(child_path, source="ga", generation=generation)
                if not child_record["execution_success"]:
                    continue

                offspring.append((child_path, child_record))
                final_success_paths.append(child_path)

                if len(final_success_paths) >= self.target_num_paths:
                    break

            if not offspring:
                fallback_attempts = 0
                while fallback_attempts < 5 and not offspring:
                    fallback_attempts += 1
                    fallback_path = self._generate_one_llm_path()
                    if fallback_path is None:
                        continue

                    fallback_record = self._evaluate_path(fallback_path, source="ga_fallback", generation=generation)
                    if fallback_record["execution_success"]:
                        offspring.append((fallback_path, fallback_record))
                        final_success_paths.append(fallback_path)
                if not offspring:
                    break

            merged = population + offspring
            merged.sort(key=lambda item: item[1]["fitness"], reverse=True)
            population = merged[: self.ga_population_size]

        save_root = Path(self.save_root_dir)
        save_root.mkdir(parents=True, exist_ok=True)

        pkl_path = save_root / f"{self.task.question_id}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(final_success_paths, f)

        print(
            f"Question ID: {self.task.question_id} done, "
            f"seed_success={sum(1 for r in self.generated_records if r['source'] == 'llm_seed' and r['execution_success'])}, "
            f"final_success={len(final_success_paths)}"
        )
