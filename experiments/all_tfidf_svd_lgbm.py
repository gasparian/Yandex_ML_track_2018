import pickle
import logging
import os
import operator

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import pipeline
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

prep = pickle.load(open(config.features_path+'/all_tfidf_svd.pickle.dat', 'rb'))
splits = pickle.load(open(config.features_path+'/all_tfidf_svd_splits.pickle.dat', 'rb'))
lgbm = LGBMRegressor(**config.params)

fold, cv_score, cv_mse, cv_mae = 0, 0, 0, 0
models = {k:{'counter':v, 'model':lgbm} for k, v in prep.items()}

for train_index, test_index in splits:
    logging.info('make features...')
    train_data = pickle.load(open(config.features_path+'/train_fold_%i.pickle.dat' % fold, 'rb'))
    test_data = pickle.load(open(config.features_path+'/test_fold_%i.pickle.dat' % fold, 'rb'))

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
    mse = mean_squared_error(train['target'].loc[test_index], models[fold]['model'].predict(test_data))
    mae = mean_absolute_error(train['target'].loc[test_index], models[fold]['model'].predict(test_data))
    cv_mse += mse / config.nfolds
    cv_mae += mae / config.nfolds

    prediction = make_prediction(train.loc[test_index], test_data, models[fold]['model'])
    score = 0
    uniques = prediction['context_id'].unique()
    for Id in uniques:
        tmp = prediction[prediction['context_id'] == Id]['reply_id'].values.tolist()
        score += ndcg_at_k(tmp, len(set(tmp))) / len(uniques)
    score *= 100000
    cv_score += score / config.nfolds
    
    mse_train = mean_squared_error(train['target'].loc[train_index], models[fold]['model'].predict(train_data))
    mae_train = mean_absolute_error(train['target'].loc[train_index], models[fold]['model'].predict(train_data))
    
    logging.info('fold#%i: ndcg = %i; mse = %s(%s); mae = %s(%s)' % (fold, int(score), mse, mse_train, mae, mae_train))
    fold += 1
    del train_data; del test_data

logging.info('averaged cv score: ndcg = %i; mse = %s; mae = %s' % (int(cv_score), cv_mse, cv_mae))

prediction = pd.DataFrame()
prediction['context_id'] = test['0']
prediction['reply_id'] = test['4']
prediction['rank'] = 0
for fold in range(config.nfolds):
    hold_out_test_data = pickle.load(open(config.features_path+'/hold_test_fold_%i.pickle.dat' % fold, 'rb'))
    prediction['rank'] += - models[fold]['model'].predict(hold_out_test_data) / config.nfolds
prediction = prediction.sort_values(by=['context_id', 'rank'])
prediction[['context_id', 'reply_id']].to_csv(path+'/sub_svd_counter_all_tfidf_lgbm.tsv',header=None, index=False, sep=' ')
prediction.to_csv(path+'/sub_rank_svd_counter_all_tfidf_lgbm.tsv', sep=' ')

pickle.dump(models, open(path+'/all_tfidf_svd_lgbm.pickle.dat', 'wb'))
logging.info('submission saved!')