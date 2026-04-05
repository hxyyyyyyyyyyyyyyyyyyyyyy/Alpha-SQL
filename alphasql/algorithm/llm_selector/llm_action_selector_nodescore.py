from alphasql.algorithm.mcts.mcts_node import *
from alphasql.algorithm.mcts.mcts_action import *
from alphasql.algorithm.llm_solver.llm_guided_decision import LLMGuidedNodeScorer, LLMDecisionGate, ActionFallbackSampler
from alphasql.llm_call.prompt_factory import get_prompt
from alphasql.llm_call.openai_llm import call_openai
from alphasql.database.sql_execution import cached_execute_sql_with_timeout, format_execution_result
from typing import Dict, Any, List, Optional
from pathlib import Path
import copy
import re

ACTION_SELECTION_TEMPERATURE = 0.3
ACTION_SELECTION_LLM_KWARGS_N = 3
FALLBACK_SOFTMAX_TEMPERATURE = 1.0


class NodeScoreActionSelector:
    """
    基于LLM建议 + NodeScorer门控 + softmax回退采样的action选择器。
    注意：门控与回退都只使用NodeScorer结果，不依赖LLM的logit/probability字段。
    """

    def __init__(self, llm_kwargs: Dict[str, Any], epsilon: float = 0.0):
        self.llm_kwargs = llm_kwargs
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0.0, 1.0], got {epsilon}")
        self.epsilon = epsilon
        self.node_scorer = LLMGuidedNodeScorer(initial_score=5.0)
        self.decision_gate = LLMDecisionGate(epsilon=epsilon)
        self.fallback_sampler = ActionFallbackSampler()
        self.action_score_memory: Dict[str, float] = {}

    def get_action_key(self, action_class) -> str:
        return action_class.__name__

    def get_action_score(self, action_class) -> float:
        return float(self.action_score_memory.get(self.get_action_key(action_class), self.node_scorer.initial_score))

    def update_action_score(self, action_class, self_consistency_reward: float, all_sql_failed: bool) -> float:
        key = self.get_action_key(action_class)
        previous_score = self.action_score_memory.get(key, self.node_scorer.initial_score)
        updated_score = self.node_scorer.score(
            previous_score=previous_score,
            self_consistency_reward=self_consistency_reward,
            all_sql_failed=all_sql_failed,
        )
        self.action_score_memory[key] = updated_score
        return updated_score

    def update_scores_from_path(self, path: List[MCTSNode], self_consistency_reward: float, all_sql_failed: bool) -> None:
        for path_node in path[1:]:
            if path_node.parent_action is None:
                continue
            self.update_action_score(path_node.parent_action.__class__, self_consistency_reward, all_sql_failed)

    def get_action_description(self, action_class) -> str:
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
        if len(valid_actions) == 1:
            return valid_actions[0]

        context = self._build_context(node)
        action_options = []
        for idx, action in enumerate(valid_actions):
            action_desc = self.get_action_description(action.__class__)
            action_options.append(f"{idx + 1}. {action_desc}")

        action_options_str = "\n".join(action_options)

        if node.selected_schema_context:
            schema_summary = self._extract_table_names(node.selected_schema_context)
        else:
            schema_summary = self._extract_table_names(node.schema_context)

        prompt = get_prompt(
            template_name="action_select",
            template_args={
                "context": context,
                "schema_summary": schema_summary,
                "action_options_str": action_options_str,
            }
        )

        llm_selected_idx = -1
        new_llm_kwargs = copy.deepcopy(self.llm_kwargs)
        new_llm_kwargs["temperature"] = ACTION_SELECTION_TEMPERATURE
        new_llm_kwargs["n"] = ACTION_SELECTION_LLM_KWARGS_N
        retry_times = max(int(new_llm_kwargs.get("n", 1)), 1)

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

                selected_idx = self._parse_selected_action(valid_actions, response)
                if 0 <= selected_idx < len(valid_actions):
                    llm_selected_idx = selected_idx
                    break

                if retry_idx < retry_times - 1:
                    continue
            except Exception:
                if retry_idx < retry_times - 1:
                    continue

        action_scores = self._build_nodescore_action_scores(node, valid_actions)

        # 若LLM给出可解析action，先做基于NodeScore的门控。
        if 0 <= llm_selected_idx < len(valid_actions):
            gate_result = self.decision_gate.gate(
                node_score=action_scores[llm_selected_idx],
                score_center=self.node_scorer.initial_score,
            )
            if gate_result.accept_llm_action:
                return valid_actions[llm_selected_idx]

        # 未采纳LLM建议或解析失败，使用NodeScorer分数做softmax回退采样。
        fallback_idx = self.fallback_sampler.sample_index(
            action_scores,
            temperature=FALLBACK_SOFTMAX_TEMPERATURE,
        )
        return valid_actions[fallback_idx]

    def _build_nodescore_action_scores(self, node: MCTSNode, valid_actions: List[MCTSAction]) -> List[float]:
        has_schema_selection = any(isinstance(pn.parent_action, SchemaSelectionAction) for pn in node.path_nodes)
        has_column_values = any(isinstance(pn.parent_action, IdentifyColumnValuesAction) for pn in node.path_nodes)
        has_column_functions = any(isinstance(pn.parent_action, IdentifyColumnFunctionsAction) for pn in node.path_nodes)
        has_sql_generation = any(isinstance(pn.parent_action, SQLGenerationAction) for pn in node.path_nodes)

        scores: List[float] = []
        for action in valid_actions:
            action_class = action.__class__
            inherited_score = self.get_action_score(action_class)

            if action_class in {SQLGenerationAction, SQLRevisionAction, EndAction}:
                all_sql_failed = node.is_valid_sql_query is False
                consistency_reward = float(node.consistency_score or 0.0)
                score = self.node_scorer.score(
                    previous_score=inherited_score,
                    self_consistency_reward=consistency_reward,
                    all_sql_failed=all_sql_failed,
                )
                scores.append(score)
                continue

            if action_class == SchemaSelectionAction:
                prior_bonus = 0.3 if not has_schema_selection else -0.1
                score = inherited_score + prior_bonus
                scores.append(score)
                continue

            if action_class == IdentifyColumnValuesAction:
                prior_bonus = 0.15 if not has_column_values else -0.05
                score = inherited_score + prior_bonus
                scores.append(score)
                continue

            if action_class == IdentifyColumnFunctionsAction:
                prior_bonus = 0.15 if not has_column_functions else -0.05
                score = inherited_score + prior_bonus
                scores.append(score)
                continue

            if action_class == RaphraseQuestionAction:
                scores.append(inherited_score + 0.05)
                continue

            # 默认分：若已生成SQL，倾向结束/修订，而非继续中间分析动作。
            scores.append(inherited_score - 0.1 if has_sql_generation else inherited_score)

        return scores

    def _extract_selected_action_from_xml(self, response: str) -> Optional[str]:
        xml_blocks = re.findall(r"```xml\s*(.*?)\s*```", response, flags=re.DOTALL | re.IGNORECASE)
        candidates = xml_blocks + [response]

        for candidate in candidates:
            answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", candidate, flags=re.DOTALL | re.IGNORECASE)
            if not answer_match:
                continue

            answer = answer_match.group(1).strip()
            if not answer:
                continue

            for line in answer.splitlines():
                cleaned = line.strip()
                if cleaned:
                    return cleaned

        return None

    def _parse_selected_action(self, valid_actions: List[MCTSAction], response: str) -> int:
        def normalize_text(text: str) -> str:
            return re.sub(r"\s+", " ", text.strip()).lower()

        def compact_text(text: str) -> str:
            return re.sub(r"[^a-z0-9]", "", normalize_text(text))

        def strip_numeric_prefix(text: str) -> str:
            return re.sub(r"^\s*\d+\s*[\.)]\s*", "", text).strip()

        selected_action = self._extract_selected_action_from_xml(response)
        if not selected_action:
            return -1

        selected_action = strip_numeric_prefix(selected_action)

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

    def _build_context(self, node: MCTSNode) -> str:
        context_parts = []
        context_parts.append(f"Question: {node.original_question}")

        if node.hint:
            context_parts.append(f"Hint: {node.hint}")

        for path_node in node.path_nodes[1:]:
            if path_node.parent_action:
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
                    if path_node.sql_query:
                        db_path = str(Path(path_node.db_root_dir) / path_node.db_id / f"{path_node.db_id}.sqlite")
                        sql_execution_result = cached_execute_sql_with_timeout(db_path, path_node.sql_query)
                        if sql_execution_result.result_type.value == "success":
                            context_parts.append("  ✓ SQL Compilation: PASSED")
                            execution_result_str = format_execution_result(sql_execution_result, row_limit=2, val_length_limit=50)
                            context_parts.append(f"  Execution Result Preview:\n{execution_result_str}")
                        else:
                            context_parts.append("  ✗ SQL Compilation: FAILED")
                            context_parts.append(f"  Error: {sql_execution_result.error_message}")
                elif isinstance(path_node.parent_action, SQLRevisionAction):
                    context_parts.append(f"- Revised SQL: {path_node.revised_sql_query}")
                    if path_node.revised_sql_query:
                        db_path = str(Path(path_node.db_root_dir) / path_node.db_id / f"{path_node.db_id}.sqlite")
                        sql_execution_result = cached_execute_sql_with_timeout(db_path, path_node.revised_sql_query)
                        if sql_execution_result.result_type.value == "success":
                            context_parts.append("- SQL Compilation: PASSED")
                            execution_result_str = format_execution_result(sql_execution_result, row_limit=2, val_length_limit=50)
                            context_parts.append(f"  Execution Result Preview:\n{execution_result_str}")
                        else:
                            context_parts.append("- SQL Compilation: FAILED")
                            context_parts.append(f"  Error: {sql_execution_result.error_message}")

        return "\n".join(context_parts)

    def _extract_table_names(self, schema_context: str) -> str:
        table_names = re.findall(r"CREATE TABLE `(\w+)`", schema_context)
        return f"Tables: {', '.join(table_names)}" if table_names else "No tables found"
