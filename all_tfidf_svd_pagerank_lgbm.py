import pickle
import logging
import os
import operator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn import pipeline
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from scipy.sparse import hstack

import config
from scripts import *

path = config.path +'/data/'+os.path.basename(__file__).split('.')[0]
try:
    os.mkdir(path)
except:
    pass
from shutil import copyfile
copyfile(config.path+'/config.py', path+'/config.py')

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(path + '/train.log', mode='w'), logging.StreamHandler()])

train = pd.read_csv(config.path+'/data/train_modified.tsv', sep=' ', index_col=0)
test = pd.read_csv(config.path+'/data/test_modified.tsv', sep=' ', index_col=0)

train.fillna('', inplace=True)
test.fillna('', inplace=True)

chars_counter = TfidfVectorizer(
            analyzer=u'char', ngram_range=(2, 5), tokenizer=None,
            max_features=config.max_features, strip_accents=None, max_df=0.9, min_df=2, lowercase=False)
chars_tsvd = TruncatedSVD(n_components=config.svd_n_components, n_iter=25, random_state=42)

words_tfidf = TfidfVectorizer(
       analyzer=u'word', ngram_range=(1, 3), tokenizer=None, use_idf=True,
       max_features=config.max_features, strip_accents=None, max_df=0.9, min_df=2, lowercase=False)
words_tsvd = TruncatedSVD(n_components=config.svd_n_components, n_iter=25, random_state=42)

nfolds = round(len(train) / len(test))
lgbm = LGBMRegressor(n_estimators=500, n_jobs=-1)
cv = KFold(n_splits=nfolds, shuffle=True, random_state=42)

fold, cv_score = 0, 0
models = {i:{'counter':{'chars_counter':chars_counter, 'chars_tsvd':chars_tsvd, 'words_tfidf':words_tfidf, 'words_tsvd':words_tsvd, 
             'pagerank':{'scaler':MinMaxScaler(feature_range=(0, 1), copy=False)}, 'words_tfidf_scaler':StandardScaler(copy=False, with_mean=False, with_std=True)},
        'model':lgbm} for i in range(nfolds)}

for train_index, test_index in cv.split(train):
    logging.info('make features...')

    train_chars_data = models[fold]['counter']['chars_counter'].fit_transform(np.hstack([train[str(col)].loc[train_index] for col in config.text_cols])) 
    train_chars_data = models[fold]['counter']['chars_tsvd'].fit_transform(train_chars_data)

    train_words_data = models[fold]['counter']['words_tfidf'].fit_transform(np.hstack([train[str(col)].loc[train_index] for col in config.text_cols]))
    vocab = pd.DataFrame.from_dict(dict(sorted(models[fold]['counter']['words_tfidf'].vocabulary_.items(), key=operator.itemgetter(1), reverse=False)), orient='index')
    weights = pagerank(np.hstack([train[str(col)].loc[train_index] for col in config.text_cols]), models[fold]['counter']['words_tfidf']).reindex(vocab.index).fillna(0).values
    models[fold]['counter']['pagerank']['ranker'] = models[fold]['counter']['pagerank']['scaler'].fit_transform(weights.T)
    train_words_data = models[fold]['counter']['pagerank']['words_tfidf_scaler'].fit_transform(train_words_data.multiply(models[fold]['counter']['pagerank']['ranker']))

    train_words_data = models[fold]['counter']['words_tsvd'].fit_transform(train_words_data)
    train_data = np.hstack([train_chars_data, train_words_data])
    del train_chars_data; del train_words_data

    test_chars_data = models[fold]['counter']['chars_counter'].transform(np.hstack([train[str(col)].loc[test_index] for col in config.text_cols]))
    test_chars_data = models[fold]['counter']['chars_tsvd'].transform(test_chars_data)

    test_words_data = models[fold]['counter']['words_tfidf'].transform(np.hstack([train[str(col)].loc[test_index] for col in config.text_cols]))
    test_words_data = models[fold]['counter']['pagerank']['words_tfidf_scaler'].transform(test_words_data.multiply(models[fold]['counter']['pagerank']['ranker']))

    test_words_data = models[fold]['counter']['words_tsvd'].transform(test_words_data)
    test_data = np.hstack([test_chars_data, test_words_data])
    del test_chars_data; del test_words_data

    logging.info(train_data.shape)
    logging.info(test_data.shape)
    logging.info('fitting model...')

    if config.early_stopping:
        models[fold]['model'].fit(train_data, train['target'].loc[train_index],
                            eval_set=[(test_data, train['target'].loc[test_index])],
                            eval_metric='l2', verbose=False,
                            early_stopping_rounds=config.num_round)
    else:
        models[fold]['model'].fit(train_data, train['target'].loc[train_index])

    logging.info('make prediction...')
    prediction = make_prediction(train.loc[test_index], test_data, models[fold]['model'])
    
    score = 0
    uniques = prediction['context_id'].unique()
    for Id in uniques:
        tmp = prediction[prediction['context_id'] == Id]['reply_id'].values.tolist()
        score += ndcg_at_k(tmp, len(set(tmp))) / len(uniques)
    score *= 100000
    cv_score += score / nfolds
    
    logging.info('fold#%i: ndcg = %i' % (fold, int(score)))
    fold += 1
logging.info('averaged cv score: ndcg = %i' % (int(cv_score)))

prediction = pd.DataFrame()
prediction['context_id'] = test['0']
prediction['reply_id'] = test['4']
prediction['rank'] = 0
for fold in range(nfolds):
    test_chars_data = models[fold]['counter']['chars_counter'].transform(np.hstack([test[str(col)] for col in config.text_cols]))
    test_chars_data = models[fold]['counter']['chars_tsvd'].transform(test_chars_data)

    test_words_data = models[fold]['counter']['words_tfidf'].transform(np.hstack([test[str(col)] for col in config.text_cols]))
    test_words_data = models[fold]['counter']['pagerank']['words_tfidf_scaler'].transform(test_words_data.multiply(models[fold]['counter']['pagerank']['ranker']))

    test_words_data = models[fold]['counter']['words_tsvd'].transform(test_words_data)
    hold_out_test_data = np.hstack([test_chars_data, test_words_data])
    del test_chars_data; del test_words_data

    prediction['rank'] += - models[fold]['model'].predict(hold_out_test_data) / nfolds

prediction = prediction.sort_values(by=['context_id', 'rank'])
prediction[['context_id', 'reply_id']].to_csv(path+'/sub_svd_counter_all_tfidf_pagerank_lgbm.tsv',header=None, index=False, sep=' ')
prediction.to_csv(path+'/sub_rank_svd_counter_all_tfidf_pagerank_lgbm.tsv', sep=' ')

pickle.dump(models, open(path+'/all_tfodf_pagerank_svd_lgbm.pickle.dat', 'wb'))
logging.info('submission saved!')