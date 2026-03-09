#!/bin/bash

start_time=$(date +%s)
set -e

echo "Start running standalone LLM+GA path generation experiment"

python -m alphasql.runner.llm_genetic_runner config/llm_genetic_bird_dev_7B.yaml

echo "Standalone LLM+GA path generation experiment completed"

end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"