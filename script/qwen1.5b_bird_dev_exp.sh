#! /bin/bash

start_time=$(date +%s)

echo "Running MCTS..."
python -m alphasql.runner.mcts_runner config/qwen1.5b_bird_dev.yaml

end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"


