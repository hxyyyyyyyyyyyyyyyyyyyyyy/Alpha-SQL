import pickle
res = pickle.load(open('results/llm_nodescore_eps0.2/gemma-4-E4B-it/bird/dev/0.pkl', 'rb'))
print(len(res))