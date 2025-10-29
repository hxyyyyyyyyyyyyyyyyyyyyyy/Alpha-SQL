conda activate vllm
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --served-model-name Qwen/Qwen2.5-Coder-7B-Instruct --port 9999 -tp 1

CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --served-model-name Qwen/Qwen2.5-Coder-32B-Instruct --port 9999 -tp 2

find . -type d -name "lsh_index" -exec rm -rf {} +

bash script/qwen1.5b_bird_dev_exp.sh > ./log/qwen1.5b.log