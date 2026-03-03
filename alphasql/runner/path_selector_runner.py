from concurrent.futures import ProcessPoolExecutor
import json
import random
import traceback
from pathlib import Path
from typing import Union
import pickle

from dotenv import load_dotenv
from tqdm import tqdm
import yaml

from alphasql.algorithm.path_selector.path_guided_solver import PathGuidedSolver
from alphasql.algorithm.path_selector.random_path_selector import RandomPathSelector
from alphasql.config.path_selector_config import PathSelectorConfig
from alphasql.llm_call.openai_llm import DEFAULT_COST_RECORDER
from alphasql.runner.task import Task

load_dotenv(override=True)


class PathSelectorRunner:
    def __init__(self, config: Union[PathSelectorConfig, str]):
        if isinstance(config, str):
            config_path = Path(config)
            assert config_path.exists(), f"Config file {config_path} does not exist"
            if config_path.suffix == ".json":
                self.config = PathSelectorConfig.model_validate_json(config_path.read_text())
            elif config_path.suffix == ".yaml":
                self.config = PathSelectorConfig.model_validate(yaml.safe_load(config_path.read_text()))
            else:
                raise ValueError(f"Unsupported config file extension: {config_path.suffix}")
        else:
            self.config = config

        if self.config.summary_paths_file is None and self.config.probability_summary_file is None:
            raise ValueError("Either summary_paths_file or probability_summary_file must be provided")

        save_root = Path(self.config.save_root_dir)
        if not save_root.exists():
            save_root.mkdir(parents=True, exist_ok=True)

        random.seed(self.config.random_seed)

    def _build_selector(self, task_id: int) -> RandomPathSelector:
        task_seed = None if self.config.random_seed is None else self.config.random_seed + int(task_id)
        return RandomPathSelector(
            summary_paths_file=self.config.summary_paths_file,
            probability_summary_file=self.config.probability_summary_file,
            random_seed=task_seed,
        )

    def run_one_task(self, task: Task):
        solver = PathGuidedSolver(
            db_root_dir=self.config.db_root_dir,
            task=task,
            max_depth=self.config.max_depth,
            save_root_dir=self.config.save_root_dir,
            llm_kwargs=self.config.mcts_model_kwargs,
            path_selector=self._build_selector(task.question_id),
            num_paths=self.config.num_reasoning_paths,
            max_path_steps=self.config.max_path_steps,
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

        with open(Path(self.config.save_root_dir) / "config.json", "w") as f:
            print(f"Saving config to {Path(self.config.save_root_dir) / 'config.json'}")
            json.dump(self.config.model_dump(), f, indent=4)

        print(f"There are {len(tasks)} tasks to solve")
        print("Using Path Selector guided solver")
        with ProcessPoolExecutor(max_workers=self.config.n_processes) as executor:
            list(tqdm(executor.map(self.run_one_task, tasks), total=len(tasks), desc="Solving tasks"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m alphasql.runner.path_selector_runner <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    runner = PathSelectorRunner(config=config_path)
    runner.run_all_tasks()
