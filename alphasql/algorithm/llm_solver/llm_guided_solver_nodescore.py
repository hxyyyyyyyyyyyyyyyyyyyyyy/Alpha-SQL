from alphasql.algorithm.mcts.mcts_node import *
from alphasql.algorithm.mcts.mcts_action import *
from alphasql.algorithm.mcts.reward import *
from alphasql.algorithm.llm_solver.llm_guided_decision import LLMGuidedNodeScorer
from alphasql.algorithm.llm_selector.llm_action_selector_nodescore import NodeScoreActionSelector
from alphasql.runner.task import Task
from alphasql.database.utils import build_table_ddl_statement
from pathlib import Path
from typing import Dict, Any, List
import pickle


class LLMGuidedNodeScoreSolver:
    """
    基于llm_guided_solver复制实现：
    - 使用NodeScoreActionSelector进行动作选择
    - 在终止节点写入node_score
    """

    def __init__(self,
                 db_root_dir: str,
                 task: Task,
                 max_steps: int,
                 max_depth: int,
                 save_root_dir: str,
                 llm_kwargs: Dict[str, Any],
                 epsilon: float,
                 reward_model: RewardModel,
                 num_paths: int = 1):
        self.llm_kwargs = llm_kwargs
        self.reward_model = reward_model
        self.task = task
        self.db_root_dir = db_root_dir
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.save_root_dir = save_root_dir
        self.num_paths = num_paths
        self.epsilon = epsilon

        self.action_selector = NodeScoreActionSelector(llm_kwargs, epsilon=epsilon)
        self.node_scorer = LLMGuidedNodeScorer(initial_score=5.0)

    def generate_one_path(self, root_node: MCTSNode) -> List[MCTSNode]:
        current = root_node
        path = [root_node]

        step = 0
        while not current.is_terminal() and step < self.max_steps and current.depth < self.max_depth:
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
            step += 1

        if not current.is_terminal():
            if current.node_type in [MCTSNodeType.SQL_GENERATION, MCTSNodeType.SQL_REVISION]:
                end_action = EndAction()
                end_nodes = end_action.create_children_nodes(current, self.llm_kwargs)
                if end_nodes:
                    current = end_nodes[0]
                    path.append(current)

        return path

    def solve(self):
        schema_context = "\n".join([build_table_ddl_statement(
            self.task.table_schema_dict[table_name].to_dict(),
            add_value_description=True,
            add_column_description=True,
            add_value_examples=True,
            add_expanded_column_name=True
        ) for table_name in self.task.table_schema_dict])

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
            table_schema_dict=self.task.table_schema_dict
        )
        root_node.path_nodes = [root_node]

        all_reasoning_paths = []

        for path_idx in range(self.num_paths):
            try:
                path = self.generate_one_path(root_node)
                if path[-1].is_terminal():
                    terminal_node = path[-1]
                    parent_sql_node = terminal_node.parent_node
                    all_sql_failed = bool(parent_sql_node and parent_sql_node.is_valid_sql_query is False)
                    self_consistency_reward = 0.0
                    if not all_sql_failed:
                        try:
                            self_consistency_reward = float(self.reward_model.get_reward(terminal_node))
                        except Exception:
                            self_consistency_reward = float(parent_sql_node.consistency_score or 0.0) if parent_sql_node else 0.0

                    terminal_action_class = parent_sql_node.parent_action.__class__ if (parent_sql_node and parent_sql_node.parent_action) else EndAction
                    previous_terminal_score = self.action_selector.get_action_score(terminal_action_class)

                    terminal_node.node_score = self.node_scorer.score(
                        previous_score=previous_terminal_score,
                        self_consistency_reward=self_consistency_reward,
                        all_sql_failed=all_sql_failed,
                    )

                    # 将rollout执行反馈写回selector记忆，用于后续rollout继承分数。
                    self.action_selector.update_scores_from_path(
                        path=path,
                        self_consistency_reward=self_consistency_reward,
                        all_sql_failed=all_sql_failed,
                    )

                    all_reasoning_paths.append(path)
                else:
                    print(f"✗ Path {path_idx + 1} did not reach terminal node")

            except Exception as e:
                print(f"✗ Error generating path {path_idx + 1}: {e}")

        save_path = Path(self.save_root_dir) / f"{self.task.question_id}.pkl"
        print(f"Question ID: {self.task.question_id} done, valid reasoning paths: {len(all_reasoning_paths)}")

        if all_reasoning_paths != []:
            with open(save_path, "wb") as f:
                pickle.dump(all_reasoning_paths, f)
