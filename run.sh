bash script/qwen7b_bird_dev_exp.sh > log/qwen7b_bird_dev_exp.log 2>&1

bash script/qwen32b_bird_dev_exp.sh > log/qwen32b_bird_dev_exp.log 2>&1

bash script/llm_guided_bird_dev_exp_7B.sh > log/qwen7b_bird_dev_exp_llm.log 2>&1

bash script/llm_guided_bird_dev_exp_7B_test.sh > log/qwen7b_bird_dev_exp_llm.log 2>&1

bash tools/sql_selection_all_results.sh

bash tools/run_eval.sh results/llm_guide24_eps0.2_chess_0

python tools/summarize_data.py --results-dir results/llm_guide24 --generate-summary-paths

python tools/copy_chess_pkls.py --source-root results/llm_guide24_eps0.5 --destination-root results/llm_guide24_eps0.5_chess --results-dir results/llm_guide24 --generate-summary-paths --force