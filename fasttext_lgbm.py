import pickle
import logging
import os

from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

import config

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

train_fasttext = np.hstack([pickle.load(open(config.path+'/data/train_text_%i_vectors.pickle.dat' % col, 'rb')) for col in config.text_cols])
test_fasttext = np.hstack([pickle.load(open(config.path+'/data/test_text_%i_vectors.pickle.dat' % col, 'rb')) for col in config.text_cols])

train.fillna('', inplace=True)
test.fillna('', inplace=True)

train['rank'] -= 1
train['target'] = train['rank']*train['7']

try:
    splits = pickle.load(open(config.features_path+'/all_tfidf_svd_splits.pickle.dat', 'rb'))
except:
    cv = KFold(n_splits=config.nfolds, shuffle=True, random_state=42)
    splits = list(cv.split(train))

params = {
    'n_estimators':100,
    'learning_rate': 0.1,
    'num_leaves':20, 
    'min_data_in_leaf':50,
    'max_bin':256
}

lgbm = LGBMRegressor(**params)

fold, cv_score, cv_mse, cv_mae = 0, 0, 0, 0
models = {i:{'model':lgbm} for i in range(config.nfolds)}

for train_index, test_index in splits:

    logging.info(train_fasttext[train_index].shape)
    logging.info(train_fasttext[test_index].shape)
    
    logging.info('fitting model...')
    if config.early_stopping:
        models[fold]['model'].fit(train_fasttext[train_index], train['target'].loc[train_index],
                            eval_set=[(train_fasttext[test_index], train['target'].loc[test_index])],
                            eval_metric='l2', verbose=False,
                            early_stopping_rounds=config.num_round)
    else:
        models[fold]['model'].fit(train_fasttext[train_index], train['target'].loc[train_index])    

    logging.info('make prediction...')
    mse = mean_squared_error(train['target'].loc[test_index], models[fold]['model'].predict(train_fasttext[test_index]))
    mae = mean_absolute_error(train['target'].loc[test_index], models[fold]['model'].predict(train_fasttext[test_index]))
    cv_mse += mse / config.nfolds
    cv_mae += mae / config.nfolds

    prediction = make_prediction(train.loc[test_index], train_fasttext[test_index], models[fold]['model'])
    score = 0
    uniques = prediction['context_id'].unique()
    for Id in uniques:
        tmp = prediction[prediction['context_id'] == Id]['reply_id'].values.tolist()
        score += ndcg_at_k(tmp, len(set(tmp))) / len(uniques)
    score *= 100000
    cv_score += score / config.nfolds

    mse_train = mean_squared_error(train['target'].loc[train_index], models[fold]['model'].predict(train_fasttext[train_index]))
    mae_train = mean_absolute_error(train['target'].loc[train_index], models[fold]['model'].predict(train_fasttext[train_index]))
    
    logging.info('fold#%i: ndcg = %i; mse = %s(%s); mae = %s(%s)' % (fold, int(score), mse, mse_train, mae, mae_train))
    fold += 1

logging.info('averaged cv score: ndcg = %i; mse = %s; mae = %s' % (int(cv_score), cv_mse, cv_mae))

prediction = pd.DataFrame()
prediction['context_id'] = test['0']
prediction['reply_id'] = test['4']
prediction['rank'] = 0
for fold in range(config.nfolds):
    prediction['rank'] += - models[fold]['model'].predict(test_fasttext) / config.nfolds
prediction = prediction.sort_values(by=['context_id', 'rank'])
prediction[['context_id', 'reply_id']].to_csv(path+'/sub_fasttext_lgbm.tsv',header=None, index=False, sep=' ')
prediction.to_csv(path+'/sub_rank_fasttext_lgbm.tsv', sep=' ')

pickle.dump(models, open(path+'/fasttext_lgbm.pickle.dat', 'wb'))
logging.info('submission saved!')