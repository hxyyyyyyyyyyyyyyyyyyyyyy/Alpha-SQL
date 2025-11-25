#! /bin/bash

start_time=$(date +%s)

echo "Running MCTS..."
python -u -m alphasql.runner.mcts_runner config/qwen7b_bird_dev.yaml

end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"


