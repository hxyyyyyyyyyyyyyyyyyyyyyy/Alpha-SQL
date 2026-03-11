from alphasql.algorithm.mcts.mcts_node import *
from alphasql.algorithm.mcts.mcts_action import *
from alphasql.llm_call.prompt_factory import get_prompt
from alphasql.llm_call.openai_llm import call_openai
from alphasql.database.sql_execution import cached_execute_sql_with_timeout, format_execution_result
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import re

ACTION_SELECTION_TEMPERATURE = 0.3
ACTION_SELECTION_LLM_KWARGS_N = 3

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
        if len(valid_actions) == 1:
            return valid_actions[0]

        # 构建当前状态的上下文
        context = self._build_context(node)
        
        # 构建可选action列表
        action_options = []
        for idx, action in enumerate(valid_actions):
            action_desc = self.get_action_description(action.__class__)
            action_options.append(f"{idx + 1}. {action_desc}")
        
        action_options_str = "\n".join(action_options)
        
        if node.selected_schema_context:
            # 如果已经做过schema selection，提取表名
            schema_summary = self._extract_table_names(node.selected_schema_context)
        else:
            schema_summary = self._extract_table_names(node.schema_context)
        
        # 构建提示词
        prompt = get_prompt(
            template_name="action_select",
            template_args={
                "context": context,
                "schema_summary": schema_summary,
                "action_options_str": action_options_str,
            }
        )
        
        new_llm_kwargs = copy.deepcopy(self.llm_kwargs)
        new_llm_kwargs["temperature"] = ACTION_SELECTION_TEMPERATURE
        new_llm_kwargs["n"] = ACTION_SELECTION_LLM_KWARGS_N
        retry_times = max(int(new_llm_kwargs.get("n", 1)), 1)

        # 使用配置文件中的模型，如果没有指定则抛出异常
        model = self.llm_kwargs.get("model")
        if not model:
            raise ValueError("Model not specified in llm_kwargs")

        for retry_idx in range(retry_times):
            try:
                
                response = call_openai(
                    prompt=prompt,
                    model=model,
                    temperature=new_llm_kwargs.get("temperature", 0.3),
                    max_tokens=new_llm_kwargs.get("max_tokens", 512),
                    n=1
                )[0]

                # 解析响应
                selected_idx = self._parse_selected_action(valid_actions, response)

                # 验证选择的索引
                if 0 <= selected_idx < len(valid_actions):
                    return valid_actions[selected_idx]
                else:
                    print("[LLM Action Selection] Error Response")
                    print(f"  prompt: {prompt}")
                    print(f"  response: {response}")
                    print(f"  valid_actions: {self._format_valid_actions(valid_actions)}")
                    print(f"  path_info: {self._format_path_info(node)}")

                if retry_idx < retry_times - 1:
                    continue

            except Exception as e:
                print("[LLM Action Selection] Error")
                print(f"  error: {e}")
                print(f"  valid_actions: {self._format_valid_actions(valid_actions)}")
                print(f"  path_info: {self._format_path_info(node)}")
                if retry_idx < retry_times - 1:
                    continue

        # 重试耗尽后，返回默认策略
        print("[LLM Action Selection] Failed to parse response after retries, using default strategy")
        return self._default_action_selection(node, valid_actions)

    def _parse_selected_action_idx(self, response: str) -> int:
        # 1) 优先解析 ```json ... ``` 代码块
        json_match = re.search(r"```json\n(.*?)```(.*?)", response, flags=re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                selected_action_number = int(result["selected_action_number"])
                return selected_action_number - 1
            except Exception:
                pass

        return -1
    
    def _parse_selected_action(self, valid_actions: List[MCTSAction], response: str) -> int:
        def normalize_text(text: str) -> str:
            return re.sub(r"\s+", " ", text.strip()).lower()

        def compact_text(text: str) -> str:
            return re.sub(r"[^a-z0-9]", "", normalize_text(text))

        def parse_json_payload(raw_response: str) -> Optional[Dict[str, Any]]:
            json_block_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_response, flags=re.DOTALL)
            if json_block_match:
                try:
                    return json.loads(json_block_match.group(1))
                except Exception:
                    pass

            raw_json_match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
            if raw_json_match:
                try:
                    return json.loads(raw_json_match.group(0))
                except Exception:
                    pass

            return None

        payload = parse_json_payload(response)
        if not payload:
            return -1

        selected_action = payload.get("selected_action")
        if not isinstance(selected_action, str):
            return -1

        selected_normalized = normalize_text(selected_action)
        selected_name = normalize_text(selected_action.split(":", 1)[0])
        selected_compact = compact_text(selected_action)
        selected_name_compact = compact_text(selected_action.split(":", 1)[0])

        for idx, action in enumerate(valid_actions):
            action_desc = self.get_action_description(action.__class__)
            action_desc_normalized = normalize_text(action_desc)
            action_name_normalized = normalize_text(action_desc.split(":", 1)[0])
            action_class_normalized = normalize_text(action.__class__.__name__)
            action_desc_compact = compact_text(action_desc)
            action_name_compact = compact_text(action_desc.split(":", 1)[0])
            action_class_compact = compact_text(action.__class__.__name__)

            if selected_normalized in {action_desc_normalized, action_name_normalized, action_class_normalized}:
                return idx

            if selected_name in {action_desc_normalized, action_name_normalized, action_class_normalized}:
                return idx

            if selected_compact in {action_desc_compact, action_name_compact, action_class_compact}:
                return idx

            if selected_name_compact in {action_desc_compact, action_name_compact, action_class_compact}:
                return idx

        return -1
        

    def _format_valid_actions(self, valid_actions: List[MCTSAction]) -> List[str]:
        return [self.get_action_description(action.__class__) for action in valid_actions]

    def _format_path_info(self, node: MCTSNode) -> Dict[str, Any]:
        executed_actions = [
            path_node.parent_action.__class__.__name__
            for path_node in node.path_nodes[1:]
            if path_node.parent_action
        ]
        return {
            "depth": node.depth,
            "executed_actions": executed_actions,
        }
    
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
                            context_parts.append(f"- SQL Compilation: PASSED")
                            execution_result_str = format_execution_result(sql_execution_result, row_limit=2, val_length_limit=50)
                            context_parts.append(f"  Execution Result Preview:\n{execution_result_str}")
                        else:
                            context_parts.append(f"- SQL Compilation: FAILED")
                            context_parts.append(f"  Error: {sql_execution_result.error_message}")
        
        # if executed_actions:
        #     context_parts.append(f"\nExecuted Actions: {', '.join(executed_actions)}")
        
        # context_parts.append(f"Current Depth: {node.depth}")
        
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
    
    def _extract_table_names(self, schema_context: str) -> str:
        """提取schema中的表名"""
        table_names = re.findall(r"CREATE TABLE `(\w+)`", schema_context)
        return f"Tables: {', '.join(table_names)}" if table_names else "No tables found"

    def _extract_shortest_table_infos(self, schema_context: str, max_tables: int = 30) -> str:
        """提取长度最短的若干个表DDL信息"""
        table_blocks = re.findall(r"(CREATE TABLE\s+`?\w+`?\s*\(.*?\);)", schema_context, flags=re.DOTALL)
        if not table_blocks:
            return self._get_schema_summary(schema_context)

        table_infos = []
        for block in table_blocks:
            name_match = re.search(r"CREATE TABLE\s+`?(\w+)`?", block)
            table_name = name_match.group(1) if name_match else "unknown_table"
            clean_block = block.strip()
            table_infos.append((len(clean_block), table_name, clean_block))

        shortest_infos = sorted(table_infos, key=lambda item: item[0])[:max_tables]
        formatted_blocks = [info[2] for info in shortest_infos]

        # return (
        #     f"Schema is truncated: only {len(formatted_blocks)} tables are provided out of {len(table_blocks)} total tables. "
        #     "This is NOT the full schema.\n"
        #     + "\n\n".join(formatted_blocks)
        # )
        
        return f"Tables: {', '.join(formatted_blocks)}\n" + "...(More tables truncated)\n" if formatted_blocks else "No tables found"
        
    
    def _get_schema_summary(self, schema_context: str) -> str:
        """获取schema的摘要信息"""
        table_count = schema_context.count("CREATE TABLE")
        return f"Total Tables: {table_count}"
