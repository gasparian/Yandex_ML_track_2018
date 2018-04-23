import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.base import ClassifierMixin, clone
# from lightgbm import LGBMRegressor
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
# import networkx as nx

# def get_model_(key, params=None):

#     models_list = {
#             'lr': SGDRegressor(),
#             'xgb': XGBRegressor(tree_method='hist', n_estimators=1000, n_jobs=4),
#             'rf': RandomForestRegressor(n_estimators=1000),
#             'lgbm': LGBMRegressor(n_estimators=1000)
#         }

#     if key == 'cat':
#         if params is not None:
#             return CatBoostRegressor(**params)
#     else:
#         if params is not None:
#             return models_list[key].set_params(**params)

# def pagerank(text, tfidf):
#     preprocessor = tfidf.build_preprocessor()
#     analyzer = tfidf.build_analyzer()
#     F = {}
#     corpus = preprocessor(text)
#     for line in corpus:
#         line = analyzer(line)
#         for i in range(len(line)-1):
#             ai, aj = line[i], line[i+1]
#             if ai not in F:
#                 F[ai] = {}
#             if aj not in F[ai]:
#                 F[ai][aj] = 0
#             F[ai][aj] += 1

#     G_all = nx.Graph()
#     for ai in F:
#         for aj in F[ai]:
#             G_all.add_edge(str(ai), str(aj))
#     pr_all = nx.pagerank(G_all, alpha=0.85)
#     return pd.DataFrame.from_dict(pr_all, orient='index')

class TqdmLoggingHandler(logging.Handler):

    def __init__ (self, level = logging.NOTSET):
        super (self.__class__, self).__init__ (level)

    def emit (self, record):
        try:
            msg = self.format (record)
            tqdm.tqdm.write (msg)
            self.flush()
        except:
            self.handleError(record)

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

def make_prediction(df, test_data, model, loss=False):
    prediction = pd.DataFrame()
    prediction['context_id'] = df['0']
    prediction['reply_id'] = df['4']
    prediction['rank'] = - model.predict(test_data)
    prediction = prediction.sort_values(by=['context_id', 'rank'])[['context_id', 'reply_id']]
    if loss:
        score = 0
        uniques = prediction['context_id'].unique()
        for Id in uniques:
            tmp = prediction[prediction['context_id'] == Id]['reply_id'].values.tolist()
            score += ndcg_at_k(tmp, k=3) / len(uniques)
        score *= 100000
        return score
    return prediction

def weighter(tfidf_data, tfidf, new_skipgram_ru, col):
    tfidf_voc = pd.DataFrame.from_dict(tfidf.vocabulary_, orient='index')
    new_skipgram_ru = new_skipgram_ru.reindex(tfidf_voc.index).reset_index()
    tfidf_voc = tfidf_voc.reset_index()
    new_skipgram_ru.fillna(0, inplace=True)
    new_skipgram_ru = new_skipgram_ru.values[:, 1:]
    new_tfidf_data = np.empty((tfidf_data.shape[0], 300))
    for i in tqdm(range(tfidf_data.shape[0]), desc='w2v col#%s'%col, ascii=True):
        idxes = np.where(tfidf_data[i].todense())[1]
        new_tfidf_data[i, :] = np.dot(tfidf_data[i, idxes].todense(), new_skipgram_ru[idxes])
    return new_tfidf_data