<<<<<<< HEAD
=======
mkdir -p log

bash script/origin_bird_dev.sh > log/origin_bird_dev.log 2>&1

bash script/llm_guided_bird.sh > log/llm_guided_bird.log 2>&1

bash script/llm_nodescore_bird.sh > log/llm_nodescore_bird_eps0.2.log 2>&1

bash script/llm_genetic_bird_dev_7B.sh > log/llm_genetic_bird_dev_7B.log 2>&1

bash tools/sql_selection.sh results/llm_guide24

bash tools/run_eval.sh results/llm_guide24_eps0.2_chess_0

python tools/summarize_data.py --results-dir results/llm_guide24 --generate-summary-paths

python tools/copy_chess_pkls.py --source-root results/llm_guide24_eps0.5 --destination-root results/llm_guide24_eps0.5_chess --results-dir results/llm_guide24 --generate-summary-paths --force

DETAIL=1 bash tools/run_eval_with_fail.sh results/origin/Qwen2.5-Coder*
>>>>>>> 7e33cc32b3460182508b75298bb1e5b654058ecc
