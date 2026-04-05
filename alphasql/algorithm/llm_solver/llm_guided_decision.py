import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence


DEFAULT_NODE_BASE_SCORE = 5.0


@dataclass
class LLMDecisionResult:
    accept_llm_action: bool
    accept_probability: float
    sampled_gate_value: float


class LLMGuidedNodeScorer:
    """Node-level scoring for execution feedback in llm-guided search."""

    def __init__(self, initial_score: float = DEFAULT_NODE_BASE_SCORE):
        self.initial_score = initial_score

    def score(
        self,
        previous_score: Optional[float],
        self_consistency_reward: Optional[float],
        all_sql_failed: bool,
    ) -> float:
        """
        Score rules with inheritance:
        - base score = previous_score (if provided) else initial_score
        - success: base + self_consistency_reward
        - all failed: base - 1
        """
        base_score = float(previous_score) if previous_score is not None else self.initial_score
        if all_sql_failed:
            return base_score - 1.0

        reward = float(self_consistency_reward or 0.0)
        return base_score + reward


class LLMDecisionGate:
    """Gate whether to trust the LLM-preselected action."""

    def __init__(self, epsilon: float = 0.0):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0.0, 1.0], got {epsilon}")
        self.epsilon = epsilon

    @staticmethod
    def sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def gate(self, node_score: float, score_center: float = DEFAULT_NODE_BASE_SCORE) -> LLMDecisionResult:
        """
        Gate policy (pure NodeScore):
        1) Convert node_score to confidence via sigmoid(node_score - score_center)
        2) Mix epsilon-greedy exploration:
           final_accept_prob = (1 - epsilon) * accept_probability
        3) Bernoulli sample from final_accept_prob.
        """
        accept_probability = self.sigmoid(float(node_score) - float(score_center))

        final_accept_prob = (1.0 - self.epsilon) * accept_probability
        sampled_gate_value = random.random()
        return LLMDecisionResult(
            accept_llm_action=sampled_gate_value < final_accept_prob,
            accept_probability=final_accept_prob,
            sampled_gate_value=sampled_gate_value,
        )


class ActionFallbackSampler:
    """Fallback action sampler with temperature softmax for exploration."""

    @staticmethod
    def softmax(scores: Sequence[float], temperature: float = 1.0) -> List[float]:
        if len(scores) == 0:
            return []
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        scaled_scores = [float(score) / temperature for score in scores]
        max_score = max(scaled_scores)
        exps = [math.exp(score - max_score) for score in scaled_scores]
        exp_sum = sum(exps)

        if exp_sum == 0:
            return [1.0 / len(scores)] * len(scores)

        return [exp_value / exp_sum for exp_value in exps]

    def sample_index(self, scores: Sequence[float], temperature: float = 1.0) -> int:
        probabilities = self.softmax(scores, temperature=temperature)
        if not probabilities:
            raise ValueError("scores cannot be empty")

        return random.choices(range(len(scores)), weights=probabilities, k=1)[0]
