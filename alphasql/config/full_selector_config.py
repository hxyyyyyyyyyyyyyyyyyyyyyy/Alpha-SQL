from pydantic import BaseModel
from typing import Any, Dict, Optional


class FullSelectorConfig(BaseModel):
    tasks_file_path: str
    subset_file_path: Optional[str]
    db_root_dir: str
    n_processes: int
    max_depth: int
    save_root_dir: str
    mcts_model_kwargs: Dict[str, Any]
    random_seed: Optional[int] = 42
    max_expansion_nodes: Optional[int] = None
