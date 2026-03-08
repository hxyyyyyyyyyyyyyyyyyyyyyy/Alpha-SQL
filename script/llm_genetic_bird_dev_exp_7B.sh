#!/bin/bash

set -e

echo "Start running standalone LLM+GA path generation experiment"

python -m alphasql.runner.llm_genetic_runner config/llm_genetic_bird_dev_7B.yaml

echo "Standalone LLM+GA path generation experiment completed"
