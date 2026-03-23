conda activate vllm

CUDA_VISIBLE_DEVICES=0,1 vllm serve /mnt/chenbang/models/Qwen3-8B --served-model-name Qwen/Qwen3-8B --port 9999 -tp 2 --max-num-seqs 128

CUDA_VISIBLE_DEVICES=2,3 vllm serve /mnt/chenbang/models/Qwen3-8B --served-model-name Qwen/Qwen3-8B --port 10000 -tp 2 --max-num-seqs 128

CUDA_VISIBLE_DEVICES=0,1 vllm serve /mnt/chenbang/models/Qwen2.5-Coder-7B-Instruct --served-model-name Qwen/Qwen2.5-Coder-7B-Instruct --port 9999 -tp 2 --max-num-seqs 64 --max-model-len 131072

CUDA_VISIBLE_DEVICES=2,3 vllm serve /mnt/chenbang/models/DeepSeek-Coder-V2-Lite-Instruct --served-model-name deepseek/DeepSeek-Coder-V2-Lite-Instruct --port 10000 -tp 2 --max-num-seqs 64 --max-model-len 131072


CUDA_VISIBLE_DEVICES=0 vllm serve /mnt/chenbang/models/Qwen2.5-Coder-7B-Instruct --served-model-name Qwen/Qwen2.5-Coder-7B-Instruct --port 9999 -tp 1 --max-model-len 131072 --hf-overrides '{"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}}'

CUDA_VISIBLE_DEVICES=1,2 vllm serve /mnt/chenbang/models/DeepSeek-Coder-V2-Lite-Instruct --served-model-name deepseek/DeepSeek-Coder-V2-Lite-Instruct --port 10000 -tp 2 --max-model-len 131072 --hf-overrides '{"rope_scaling": {"rope_type": "yarn", "factor": 40.0, "original_max_position_embeddings": 4096}}'

CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --served-model-name Qwen/Qwen2.5-Coder-7B-Instruct --port 10000 -tp 1 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072

CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --served-model-name Qwen/Qwen2.5-Coder-32B-Instruct --port 9999 -tp 2 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072

find . -type d -name "lsh_index" -exec rm -rf {} +

bash script/qwen1.5b_bird_dev_exp.sh > ./log/qwen1.5b.log

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --served-model-name Qwen/Qwen2.5-Coder-32B-Instruct --port 9999 -tp 4 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072