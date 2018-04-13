import numpy as np
import pandas as pd

def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k=3, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max 

def make_prediction(df, test_data, model):
    prediction = pd.DataFrame()
    prediction['context_id'] = df['0']
    prediction['reply_id'] = df['4']
    prediction['rank'] = - model.predict(test_data)
    prediction = prediction.sort_values(by=['context_id', 'rank'])[['context_id', 'reply_id']]
    return prediction