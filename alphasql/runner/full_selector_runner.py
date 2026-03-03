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

from alphasql.algorithm.full_selector.full_tree_solver import FullTreeSolver
from alphasql.config.full_selector_config import FullSelectorConfig
from alphasql.llm_call.openai_llm import DEFAULT_COST_RECORDER
from alphasql.runner.task import Task

load_dotenv(override=True)


class FullSelectorRunner:
    def __init__(self, config: Union[FullSelectorConfig, str]):
        if isinstance(config, str):
            config_path = Path(config)
            assert config_path.exists(), f"Config file {config_path} does not exist"
            if config_path.suffix == ".json":
                self.config = FullSelectorConfig.model_validate_json(config_path.read_text())
            elif config_path.suffix == ".yaml":
                self.config = FullSelectorConfig.model_validate(yaml.safe_load(config_path.read_text()))
            else:
                raise ValueError(f"Unsupported config file extension: {config_path.suffix}")
        else:
            self.config = config

        save_root = Path(self.config.save_root_dir)
        if not save_root.exists():
            save_root.mkdir(parents=True, exist_ok=True)

        random.seed(self.config.random_seed)

    def run_one_task(self, task: Task):
        solver = FullTreeSolver(
            db_root_dir=self.config.db_root_dir,
            task=task,
            max_depth=self.config.max_depth,
            save_root_dir=self.config.save_root_dir,
            llm_kwargs=self.config.mcts_model_kwargs,
            max_expansion_nodes=self.config.max_expansion_nodes,
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
        print("Using Full Selector solver (full tree traversal, no backpropagate)")
        with ProcessPoolExecutor(max_workers=self.config.n_processes) as executor:
            list(tqdm(executor.map(self.run_one_task, tasks), total=len(tasks), desc="Solving tasks"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m alphasql.runner.full_selector_runner <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    runner = FullSelectorRunner(config=config_path)
    runner.run_all_tasks()
