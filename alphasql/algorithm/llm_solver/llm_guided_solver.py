from alphasql.algorithm.mcts.mcts_node import *
from alphasql.algorithm.mcts.mcts_action import *
from alphasql.algorithm.mcts.reward import *
from alphasql.algorithm.llm_selector.llm_action_selector import LLMActionSelector
from alphasql.runner.task import Task
from alphasql.database.utils import build_table_ddl_statement
from pathlib import Path
from typing import Dict, Any, List
import pickle
import random


class LLMGuidedSolver:
    """
    使用大模型引导的求解器，替代MCTS搜索
    不使用蒙特卡洛树搜索，而是使用LLM直接选择每一步的最优action
    """
    
    def __init__(self,
                 db_root_dir: str,
                 task: Task, 
                 max_steps: int,
                 max_depth: int,
                 save_root_dir: str,
                 llm_kwargs: Dict[str, Any],
                 reward_model: RewardModel,
                 num_paths: int = 1):
        """
        Args:
            db_root_dir: 数据库根目录
            task: 任务对象
            max_steps: 每条路径的最大步数
            max_depth: 最大深度
            save_root_dir: 保存结果的根目录
            llm_kwargs: 大模型调用参数
            reward_model: 奖励模型
            num_paths: 生成的推理路径数量
        """
        self.llm_kwargs = llm_kwargs
        self.reward_model = reward_model
        self.task = task
        self.db_root_dir = db_root_dir
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.save_root_dir = save_root_dir
        self.num_paths = num_paths
        
        # 创建LLM action选择器
        self.action_selector = LLMActionSelector(llm_kwargs)
    
    def generate_one_path(self, root_node: MCTSNode) -> List[MCTSNode]:
        """
        使用LLM生成一条完整的推理路径
        """
        current = root_node
        path = [root_node]
        
        step = 0
        while not current.is_terminal() and step < self.max_steps and current.depth < self.max_depth:
            #print(f"  Step {step + 1}, Current node type: {current.node_type}")
            
            # 获取当前节点的有效action空间
            valid_action_space = get_valid_action_space_for_node(current)
            
            if not valid_action_space:
                print(f"  No valid actions available at depth {current.depth}")
                break
            
            # 使用LLM选择下一个action
            selected_action = self.action_selector.select_action(current, valid_action_space)
            
            # 执行选中的action，生成子节点
            children_nodes = selected_action.create_children_nodes(current, self.llm_kwargs)
            
            if not children_nodes:
                print(f"  Action {selected_action.__class__.__name__} generated no children")
                break
            
            # 选择第一个子节点继续（因为LLM已经做了选择）
            current = children_nodes[0]
            current.children = []  # 清空children，因为我们只沿着一条路径前进
            path.append(current)
            
            step += 1
        
        # 如果没有到达终止节点，尝试强制结束
        if not current.is_terminal():
            if current.node_type in [MCTSNodeType.SQL_GENERATION, MCTSNodeType.SQL_REVISION]:
                end_action = EndAction()
                end_nodes = end_action.create_children_nodes(current, self.llm_kwargs)
                if end_nodes:
                    current = end_nodes[0]
                    path.append(current)
        
        return path
    
    def solve(self):
        """
        使用LLM引导的方式求解任务
        """
        # 构建schema上下文
        schema_context = "\n".join([build_table_ddl_statement(
            self.task.table_schema_dict[table_name].to_dict(), 
            add_value_description=True,
            add_column_description=True,
            add_value_examples=True,
            add_expanded_column_name=True
        ) for table_name in self.task.table_schema_dict])
        
        # 创建根节点
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
        
        # 生成多条推理路径
        for path_idx in range(self.num_paths):
            #print(f"\n{'='*80}")
            #print(f"Question ID: {self.task.question_id}, Generating path {path_idx + 1} / {self.num_paths}")
            #print(f"{'='*80}")
            
            try:
                path = self.generate_one_path(root_node)
                
                # 检查路径是否有效（是否到达了终止节点）
                if path[-1].is_terminal():
                    all_reasoning_paths.append(path)
                    print(f"✓ Path {path_idx + 1} completed successfully")
                    print(f"  Final SQL: {path[-1].final_sql_query}")
                else:
                    print(f"✗ Path {path_idx + 1} did not reach terminal node")
                    
            except Exception as e:
                print(f"✗ Error generating path {path_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
        
        # 保存结果
        save_path = Path(self.save_root_dir) / f"{self.task.question_id}.pkl"
        print(f"\n{'='*80}")
        print(f"Question ID: {self.task.question_id} completed")
        print(f"Successfully generated {len(all_reasoning_paths)} reasoning paths")
        print(f"Saving to {save_path}")
        print(f"{'='*80}\n")
        
        with open(save_path, "wb") as f:
            pickle.dump(all_reasoning_paths, f)
