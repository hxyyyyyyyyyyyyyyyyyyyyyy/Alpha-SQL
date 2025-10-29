from alphasql.algorithm.mcts.mcts_node import *
from alphasql.algorithm.mcts.mcts_action import *
from alphasql.llm_call.prompt_factory import get_prompt
from alphasql.llm_call.openai_llm import call_openai
from alphasql.database.sql_execution import cached_execute_sql_with_timeout, format_execution_result
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import re


class LLMActionSelector:
    """
    使用大模型进行action选择，替代MCTS的UCB选择策略
    """
    
    def __init__(self, llm_kwargs: Dict[str, Any]):
        self.llm_kwargs = llm_kwargs
    
    def get_action_description(self, action_class) -> str:
        """获取action的描述信息"""
        action_descriptions = {
            RaphraseQuestionAction: "Rephrase Question: Reformulate the question to make it clearer and more specific",
            SchemaSelectionAction: "Schema Selection: Select relevant tables and columns from the database schema",
            IdentifyColumnValuesAction: "Identify Column Values: Identify specific column values mentioned in the question",
            IdentifyColumnFunctionsAction: "Identify Column Functions: Identify SQL functions (COUNT, SUM, AVG, etc.) needed for the query",
            SQLGenerationAction: "SQL Generation: Generate the final SQL query based on previous analysis",
            SQLRevisionAction: "SQL Revision: Revise the previously generated SQL query based on execution results",
            EndAction: "End: Finish the current reasoning path"
        }
        result = action_descriptions.get(action_class, action_class.__name__)
        return result if result else action_class.__name__
    
    def select_action(self, node: MCTSNode, valid_actions: List[MCTSAction]) -> MCTSAction:
        """
        使用大模型选择最优的下一步action
        """
        # 构建当前状态的上下文
        context = self._build_context(node)
        
        # 构建可选action列表
        action_options = []
        for idx, action in enumerate(valid_actions):
            action_desc = self.get_action_description(action.__class__)
            action_options.append(f"{idx + 1}. {action_desc}")
        
        action_options_str = "\n".join(action_options)
        
        # 获取schema信息 - 若已进行SchemaSelection则使用selected_schema，否则使用完整schema
        schema_context = node.selected_schema_context if node.selected_schema_context else node.schema_context
        
        # 构建提示词
        prompt = f"""You are an expert SQL query generator. Given the current state of SQL query generation process, select the most appropriate next action.

Database Schema:
{schema_context}

Current State:
{context}

Available Actions:
{action_options_str}

Please analyze the current state and choose the best next action. Consider:
1. Have we done enough preparation to generate SQL? If done, feel free to generate SQL directly, If not, choose actions that help gather more information.
2. What information is still missing?
3. What is the logical next step in the SQL generation process?
4. If SQL has been generated, check its compilation status to decide next steps.

Respond in the following JSON format:
```json
{{
    "reasoning": "Your reasoning for choosing this action",
    "selected_action_number": <number between 1 and {len(valid_actions)}>
}}
```
"""
        
        # 调用大模型
        try:
            # 使用配置文件中的模型，如果没有指定则抛出异常
            model = self.llm_kwargs.get("model")
            if not model:
                raise ValueError("Model not specified in llm_kwargs")
            
            response = call_openai(
                prompt=prompt,
                model=model,
                temperature=self.llm_kwargs.get("temperature", 0.3),
                max_tokens=self.llm_kwargs.get("max_tokens", 512),
                n=1
            )[0]
            
            # 解析响应
            print(f"\n[LLM Action Selection] Prompt: {prompt}")
            json_match = re.search(r"```json\n(.*?)```", response, flags=re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
                selected_idx = result.get("selected_action_number", 1) - 1
                reasoning = result.get("reasoning", "")
                
                # 验证选择的索引
                if 0 <= selected_idx < len(valid_actions):
                    print(f"\n[LLM Action Selection] Reasoning: {reasoning}")
                    print(f"[LLM Action Selection] Selected: {self.get_action_description(valid_actions[selected_idx].__class__)}")
                    return valid_actions[selected_idx]
            
            # 如果解析失败，返回默认策略
            print("[LLM Action Selection] Failed to parse response, using default strategy")
            return self._default_action_selection(node, valid_actions)
            
        except Exception as e:
            print(f"[LLM Action Selection] Error: {e}, using default strategy")
            return self._default_action_selection(node, valid_actions)
    
    def _build_context(self, node: MCTSNode) -> str:
        """构建当前节点的上下文信息"""
        context_parts = []
        
        context_parts.append(f"Question: {node.original_question}")
        
        if node.hint:
            context_parts.append(f"Hint: {node.hint}")
        
        # 收集路径中已执行的actions
        executed_actions = []
        for path_node in node.path_nodes[1:]:  # 跳过root节点
            if path_node.parent_action:
                action_name = path_node.parent_action.__class__.__name__
                executed_actions.append(action_name)
                
                if isinstance(path_node.parent_action, RaphraseQuestionAction):
                    context_parts.append(f"- Rephrased Question: {path_node.rephrased_question}")
                elif isinstance(path_node.parent_action, SchemaSelectionAction):
                    if path_node.selected_schema_dict:
                        context_parts.append(f"- Selected Schema: {len(path_node.selected_schema_dict)} tables")
                elif isinstance(path_node.parent_action, IdentifyColumnValuesAction):
                    if path_node.identified_column_values:
                        context_parts.append(f"- Identified Column Values: {path_node.identified_column_values[:100]}...")
                elif isinstance(path_node.parent_action, IdentifyColumnFunctionsAction):
                    if path_node.identified_column_functions:
                        context_parts.append(f"- Identified Functions: {path_node.identified_column_functions[:100]}...")
                elif isinstance(path_node.parent_action, SQLGenerationAction):
                    context_parts.append(f"- Generated SQL: {path_node.sql_query}")
                    # 添加SQL执行状态信息
                    if path_node.sql_query:
                        db_path = str(Path(path_node.db_root_dir) / path_node.db_id / f"{path_node.db_id}.sqlite")
                        sql_execution_result = cached_execute_sql_with_timeout(db_path, path_node.sql_query)
                        if sql_execution_result.result_type.value == "success":
                            context_parts.append(f"  ✓ SQL Compilation: PASSED")
                            execution_result_str = format_execution_result(sql_execution_result, row_limit=2, val_length_limit=50)
                            context_parts.append(f"  Execution Result Preview:\n{execution_result_str}")
                        else:
                            context_parts.append(f"  ✗ SQL Compilation: FAILED")
                            context_parts.append(f"  Error: {sql_execution_result.error_message}")
                elif isinstance(path_node.parent_action, SQLRevisionAction):
                    context_parts.append(f"- Revised SQL: {path_node.revised_sql_query}")
                    # 添加修订后SQL的执行状态信息
                    if path_node.revised_sql_query:
                        db_path = str(Path(path_node.db_root_dir) / path_node.db_id / f"{path_node.db_id}.sqlite")
                        sql_execution_result = cached_execute_sql_with_timeout(db_path, path_node.revised_sql_query)
                        if sql_execution_result.result_type.value == "success":
                            context_parts.append(f"  ✓ SQL Compilation: PASSED")
                            execution_result_str = format_execution_result(sql_execution_result, row_limit=2, val_length_limit=50)
                            context_parts.append(f"  Execution Result Preview:\n{execution_result_str}")
                        else:
                            context_parts.append(f"  ✗ SQL Compilation: FAILED")
                            context_parts.append(f"  Error: {sql_execution_result.error_message}")
        
        if executed_actions:
            context_parts.append(f"\nExecuted Actions: {', '.join(executed_actions)}")
        
        context_parts.append(f"Current Depth: {node.depth}")
        
        return "\n".join(context_parts)
    
    def _default_action_selection(self, node: MCTSNode, valid_actions: List[MCTSAction]) -> MCTSAction:
        """默认的action选择策略（启发式规则）"""
        # 优先级策略：
        # 1. 如果可以生成SQL且已有足够信息，优先生成SQL
        # 2. 如果还没有schema selection，优先做schema selection
        # 3. 如果已有SQL，优先End
        # 4. 其他情况选择第一个可用action
        
        action_classes = [action.__class__ for action in valid_actions]
        
        # 检查是否已经有SQL生成
        has_sql_generation = any(isinstance(pn.parent_action, SQLGenerationAction) for pn in node.path_nodes)
        has_schema_selection = any(isinstance(pn.parent_action, SchemaSelectionAction) for pn in node.path_nodes)
        
        if EndAction in action_classes and has_sql_generation:
            return next(a for a in valid_actions if isinstance(a, EndAction))
        
        if SQLGenerationAction in action_classes and has_schema_selection:
            return next(a for a in valid_actions if isinstance(a, SQLGenerationAction))
        
        if SchemaSelectionAction in action_classes and not has_schema_selection:
            return next(a for a in valid_actions if isinstance(a, SchemaSelectionAction))
        
        return valid_actions[0]
