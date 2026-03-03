from pathlib import Path
from typing import Any, Dict, List, Optional
import pickle

from alphasql.algorithm.mcts.mcts_action import (
    EndAction,
    IdentifyColumnFunctionsAction,
    IdentifyColumnValuesAction,
    MCTSAction,
    MCTSNodeType,
    RaphraseQuestionAction,
    SQLGenerationAction,
    SQLRevisionAction,
    SchemaSelectionAction,
)
from alphasql.algorithm.mcts.mcts_node import MCTSNode, get_valid_action_space_for_node
from alphasql.algorithm.path_selector.random_path_selector import RandomPathSelector
from alphasql.algorithm.path_selector.transition_preprocessor import normalize_node_name
from alphasql.database.utils import build_table_ddl_statement
from alphasql.runner.task import Task


NODE_NAME_TO_ACTION = {
    "REPHRASE_QUESTION": RaphraseQuestionAction,
    "SCHEMA_SELECTION": SchemaSelectionAction,
    "IDENTIFY_COLUMN_VALUES": IdentifyColumnValuesAction,
    "IDENTIFY_COLUMN_FUNCTIONS": IdentifyColumnFunctionsAction,
    "SQL_GENERATION": SQLGenerationAction,
    "SQL_REVISION": SQLRevisionAction,
    "END": EndAction,
}


class PathGuidedSolver:
    def __init__(
        self,
        db_root_dir: str,
        task: Task,
        max_depth: int,
        save_root_dir: str,
        llm_kwargs: Dict[str, Any],
        path_selector: RandomPathSelector,
        num_paths: int = 1,
        max_path_steps: int = 16,
    ):
        self.db_root_dir = db_root_dir
        self.task = task
        self.max_depth = max_depth
        self.save_root_dir = save_root_dir
        self.llm_kwargs = llm_kwargs
        self.path_selector = path_selector
        self.num_paths = num_paths
        self.max_path_steps = max_path_steps

    def _build_root_node(self) -> MCTSNode:
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

    def _match_action(self, target_node_name: str, valid_actions: List[MCTSAction]) -> Optional[MCTSAction]:
        target_action_cls = NODE_NAME_TO_ACTION.get(normalize_node_name(target_node_name))
        if target_action_cls is None:
            return None
        for action in valid_actions:
            if isinstance(action, target_action_cls):
                return action
        return None

    def _execute_template_path(self, root_node: MCTSNode, sampled_path: List[str]) -> Optional[List[MCTSNode]]:
        current = root_node
        path_nodes = [root_node]

        normalized_path = [normalize_node_name(node_name) for node_name in sampled_path]
        if not normalized_path or normalized_path[0] != "ROOT":
            normalized_path = ["ROOT"] + normalized_path

        for next_node_name in normalized_path[1:]:
            if current.is_terminal() or current.depth >= self.max_depth:
                break

            valid_actions = get_valid_action_space_for_node(current)
            if not valid_actions:
                break

            selected_action = self._match_action(next_node_name, valid_actions)
            if selected_action is None:
                break

            children_nodes = selected_action.create_children_nodes(current, self.llm_kwargs)
            if not children_nodes:
                break

            current = children_nodes[0]
            current.children = []
            path_nodes.append(current)

        if not current.is_terminal():
            valid_actions = get_valid_action_space_for_node(current)
            end_action = next((action for action in valid_actions if isinstance(action, EndAction)), None)
            if end_action is not None:
                end_nodes = end_action.create_children_nodes(current, self.llm_kwargs)
                if end_nodes:
                    current = end_nodes[0]
                    path_nodes.append(current)

        if path_nodes[-1].is_terminal():
            return path_nodes
        return None

    def solve(self):
        root_node = self._build_root_node()
        sampled_paths = self.path_selector.sample_paths(
            num_samples=self.num_paths,
            case_id=str(self.task.question_id),
            max_steps=self.max_path_steps,
        )

        all_reasoning_paths: List[List[MCTSNode]] = []
        for sampled_path in sampled_paths:
            try:
                executed_path = self._execute_template_path(root_node, sampled_path)
                if executed_path is not None:
                    all_reasoning_paths.append(executed_path)
            except Exception as e:
                print(f"Error executing sampled path for question {self.task.question_id}: {e}")

        save_path = Path(self.save_root_dir) / f"{self.task.question_id}.pkl"
        print(f"Question ID: {self.task.question_id} done, valid reasoning paths: {len(all_reasoning_paths)}")
        if all_reasoning_paths:
            with open(save_path, "wb") as f:
                pickle.dump(all_reasoning_paths, f)
