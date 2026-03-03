from pydantic import BaseModel
from typing import Dict, Any, Optional


class PathSelectorConfig(BaseModel):
    tasks_file_path: str
    subset_file_path: Optional[str]
    db_root_dir: str
    n_processes: int
    max_depth: int
    save_root_dir: str
    mcts_model_kwargs: Dict[str, Any]
    reward_model_kwargs: Optional[Dict[str, Any]] = None
    random_seed: Optional[int] = 42

    summary_paths_file: Optional[str] = None
    probability_summary_file: Optional[str] = None
    num_reasoning_paths: int = 1
    max_path_steps: int = 16