from pydantic import BaseModel
from typing import Dict, Any, Optional


class LLMGAConfig(BaseModel):
    tasks_file_path: str
    subset_file_path: Optional[str]
    db_root_dir: str
    n_processes: int
    max_depth: int
    max_path_steps: int = 16
    save_root_dir: str
    mcts_model_kwargs: Dict[str, Any]
    random_seed: Optional[int] = 42

    llm_seed_num_paths: int = 6
    llm_seed_num_retry: int = 12
    target_num_paths: int = 16

    ga_population_size: int = 8
    ga_max_generations: int = 20
    ga_crossover_rate: float = 0.7
    ga_mutation_rate: float = 0.3
    ga_tournament_size: int = 3
