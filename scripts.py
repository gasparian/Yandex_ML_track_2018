import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


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

def make_prediction(df, test_data, model):
    prediction = pd.DataFrame()
    prediction['context_id'] = df['0']
    prediction['reply_id'] = df['4']
    prediction['rank'] = - model.predict(test_data)
    prediction = prediction.sort_values(by=['context_id', 'rank'])[['context_id', 'reply_id']]
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

class EarlyStopping(ClassifierMixin):
    def __init__(self, estimator, max_n_estimators, scorer,
                 n_min_iterations=50, scale=1.02):

        self.estimator = estimator
        self.max_n_estimators = max_n_estimators
        self.scorer = scorer
        self.scale = scale
        self.n_min_iterations = n_min_iterations
    
    def _make_estimator(self, append=True):

        estimator = clone(self.estimator)
        estimator.n_estimators = 1
        estimator.warm_start = True
        return estimator
    
    def fit(self, X, y):

        est = self._make_estimator()
        self.scores_ = []

        for n_est in range(1, self.max_n_estimators+1):
            est.n_estimators = n_est
            est.fit(X,y)
            
            score = self.scorer(est)
            self.estimator_ = est
            self.scores_.append(score)

            if (n_est > self.n_min_iterations and
                score > self.scale*np.min(self.scores_)):
                return self

        return self

def es_scorer(X, y, est, scorer):
    pred = est.predict(X)
    return scorer(y, pred)

def get_model_(key, params=None):

    models_list = {
            'lr': SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15),
            'xgb': XGBRegressor(n_estimators=1000),
            'rf': RandomForestRegressor(n_estimators=1000),
            'lgbm': LGBMRegressor(n_estimators=1000)
        }

    if key == 'cat':
        if params is not None:
            return CatBoostRegressor(**params)
    else:
        if params is not None:
            return models_list[key].set_params(**params)