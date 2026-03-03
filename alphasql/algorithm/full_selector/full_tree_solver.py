from pathlib import Path
from typing import Any, Dict, List, Optional
import pickle

from alphasql.algorithm.mcts.mcts_action import EndAction, MCTSNodeType
from alphasql.algorithm.mcts.mcts_node import MCTSNode, get_valid_action_space_for_node
from alphasql.database.utils import build_table_ddl_statement
from alphasql.runner.task import Task


class FullTreeSolver:
    def __init__(
        self,
        db_root_dir: str,
        task: Task,
        max_depth: int,
        save_root_dir: str,
        llm_kwargs: Dict[str, Any],
        max_expansion_nodes: Optional[int] = None,
    ):
        self.db_root_dir = db_root_dir
        self.task = task
        self.max_depth = max_depth
        self.save_root_dir = save_root_dir
        self.llm_kwargs = llm_kwargs
        self.max_expansion_nodes = max_expansion_nodes

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

    def _expand_node(self, node: MCTSNode) -> List[MCTSNode]:
        if node.children:
            return node.children

        valid_action_space = get_valid_action_space_for_node(node)
        for action in valid_action_space:
            children_nodes = action.create_children_nodes(node, self.llm_kwargs)
            node.children.extend(children_nodes)
        return node.children

    def _try_force_end(self, node: MCTSNode) -> Optional[MCTSNode]:
        if node.node_type not in [MCTSNodeType.SQL_GENERATION, MCTSNodeType.SQL_REVISION]:
            return None

        end_nodes = EndAction().create_children_nodes(node, self.llm_kwargs)
        if not end_nodes:
            return None
        return end_nodes[0]

    def solve(self):
        root_node = self._build_root_node()

        all_reasoning_paths: List[List[MCTSNode]] = []
        stack: List[MCTSNode] = [root_node]
        expanded_nodes = 0

        while stack:
            current = stack.pop()

            if current.is_terminal():
                all_reasoning_paths.append(current.path_nodes)
                continue

            if current.depth >= self.max_depth:
                forced_end_node = self._try_force_end(current)
                if forced_end_node is not None and forced_end_node.is_terminal():
                    all_reasoning_paths.append(forced_end_node.path_nodes)
                continue

            if self.max_expansion_nodes is not None and expanded_nodes >= self.max_expansion_nodes:
                print(
                    f"Question ID: {self.task.question_id}, reached max_expansion_nodes={self.max_expansion_nodes}, stop early"
                )
                break

            children = self._expand_node(current)
            expanded_nodes += 1

            if not children:
                forced_end_node = self._try_force_end(current)
                if forced_end_node is not None and forced_end_node.is_terminal():
                    all_reasoning_paths.append(forced_end_node.path_nodes)
                continue

            for child in reversed(children):
                stack.append(child)

        save_path = Path(self.save_root_dir) / f"{self.task.question_id}.pkl"
        print(
            f"Question ID: {self.task.question_id} done, valid reasoning paths: {len(all_reasoning_paths)}, expanded nodes: {expanded_nodes}"
        )
        if all_reasoning_paths:
            with open(save_path, "wb") as f:
                pickle.dump(all_reasoning_paths, f)
