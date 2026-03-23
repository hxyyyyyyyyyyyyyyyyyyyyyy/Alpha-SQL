bash script/qwen7b_bird_dev_exp.sh > log/qwen7b_bird_dev_exp.log 2>&1

bash script/qwen32b_bird_dev_exp.sh > log/qwen32b_bird_dev_exp.log 2>&1

bash script/llm_guided_bird_dev_exp_7B.sh > log/qwen7b_bird_dev_exp_llm.log 2>&1

bash script/llm_guided_bird_dev_exp_7B_test.sh > log/qwen7b_bird_dev_exp_llm.log 2>&1

bash script/sql_selection_all_results.sh

python summarize_data.py