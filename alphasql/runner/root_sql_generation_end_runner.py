from concurrent.futures import ProcessPoolExecutor
import json
import random
import traceback
from pathlib import Path
from typing import List, Sequence, Union
import pickle

from dotenv import load_dotenv
from tqdm import tqdm
import yaml

from alphasql.algorithm.path_selector.path_guided_solver import PathGuidedSolver
from alphasql.config.mcts_config import MCTSConfig
from alphasql.llm_call.openai_llm import DEFAULT_COST_RECORDER
from alphasql.runner.task import Task

load_dotenv(override=True)


class FixedPathSelector:
    """Always returns one fixed path template for each task."""

    def __init__(self, fixed_path: Sequence[str]):
        normalized = [node.upper() for node in fixed_path]
        if not normalized or normalized[0] != "ROOT":
            normalized = ["ROOT"] + normalized
        if normalized[-1] != "END":
            normalized = normalized + ["END"]
        self.fixed_path = normalized

    def sample_paths(self, num_samples: int, case_id: str = "", max_steps: int = 16) -> List[List[str]]:
        _ = case_id
        _ = max_steps
        return [list(self.fixed_path) for _ in range(num_samples)]


class RootSqlGenerationEndRunner:
    def __init__(self, config: Union[MCTSConfig, str]):
        if isinstance(config, str):
            config_path = Path(config)
            assert config_path.exists(), f"Config file {config_path} does not exist"
            if config_path.suffix == ".json":
                self.config = MCTSConfig.model_validate_json(config_path.read_text())
            elif config_path.suffix == ".yaml":
                self.config = MCTSConfig.model_validate(yaml.safe_load(config_path.read_text()))
            else:
                raise ValueError(f"Unsupported config file extension: {config_path.suffix}")
        else:
            self.config = config

        save_root = Path(self.config.save_root_dir)
        if not save_root.exists():
            save_root.mkdir(parents=True, exist_ok=True)

        random.seed(self.config.random_seed)
        self.fixed_path_selector = FixedPathSelector(["ROOT", "SQL_GENERATION", "END"])

    def run_one_task(self, task: Task):
        solver = PathGuidedSolver(
            db_root_dir=self.config.db_root_dir,
            task=task,
            max_depth=self.config.max_depth,
            save_root_dir=self.config.save_root_dir,
            llm_kwargs=self.config.mcts_model_kwargs,
            path_selector=self.fixed_path_selector,
            num_paths=1,
            max_path_steps=3,
        )
        try:
            solver.solve()
        except Exception as e:
            print("-" * 100)
            print(f"Error solving task {task.question_id}: {e}")
            traceback.print_exc()
            print(f"The task {task.question_id} has been given up")
            print("-" * 100)
        DEFAULT_COST_RECORDER.print_profile()

    def run_all_tasks(self):
        with open(self.config.tasks_file_path, "rb") as f:
            tasks = pickle.load(f)

        if self.config.subset_file_path:
            print(f"Using subset file {self.config.subset_file_path} to filter tasks")
            with open(self.config.subset_file_path, "r") as f:
                subset_data = json.load(f)
                subset_ids = [item["question_id"] for item in subset_data]
                tasks = [task for task in tasks if task.question_id in subset_ids]
            print(f"Filtered {len(tasks)} tasks")

        done_task_ids = []
        for pkl_file in Path(self.config.save_root_dir).glob("*.pkl"):
            done_task_ids.append(int(pkl_file.stem))
        print(f"Ignore done task ids: {done_task_ids}")
        tasks = [task for task in tasks if task.question_id not in done_task_ids]

        config_payload = self.config.model_dump()
        config_payload["fixed_path"] = ["ROOT", "SQL_GENERATION", "END"]
        with open(Path(self.config.save_root_dir) / "config.json", "w") as f:
            print(f"Saving config to {Path(self.config.save_root_dir) / 'config.json'}")
            json.dump(config_payload, f, indent=4)

        print(f"There are {len(tasks)} tasks to solve")
        print("Using fixed path runner: ROOT -> SQL_GENERATION -> END")
        with ProcessPoolExecutor(max_workers=self.config.n_processes) as executor:
            list(tqdm(executor.map(self.run_one_task, tasks), total=len(tasks), desc="Solving tasks"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m alphasql.runner.root_sql_generation_end_runner <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    runner = RootSqlGenerationEndRunner(config=config_path)
    runner.run_all_tasks()
