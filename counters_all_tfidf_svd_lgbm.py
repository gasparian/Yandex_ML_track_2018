import pickle
import logging
import os
import operator

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import pipeline
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

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

counters_pipe = pipeline.FeatureUnion(
    n_jobs = -1,
    transformer_list = [
    ('chars_features', pipeline.Pipeline([
        ('chars_counter', TfidfVectorizer(
            analyzer=u'char', ngram_range=(2, 5), tokenizer=None,
            max_features=config.max_features, strip_accents=None, max_df=0.9, min_df=2, lowercase=False)),
        ('chars_tsvd', TruncatedSVD(n_components=config.svd_n_components, n_iter=25, random_state=42))])),
    ('words_features', pipeline.Pipeline([
        ('words_counter', TfidfVectorizer(
       analyzer=u'word', ngram_range=(1, 3), tokenizer=None, use_idf=True,
       max_features=config.max_features, strip_accents=None, max_df=0.9, min_df=2, lowercase=False)),
        ('words_tsvd', TruncatedSVD(n_components=config.svd_n_components, n_iter=25, random_state=42))])),
])

nfolds = round(len(train) / len(test))
lgbm = LGBMRegressor(n_estimators=500, n_jobs=-1)
cv = KFold(n_splits=nfolds, shuffle=True, random_state=42)

fold, cv_score = 0, 0
models = {i:{'counter':counters_pipe, 'model':lgbm} for i in range(nfolds)}

for train_index, test_index in cv.split(train):
    logging.info('make features...')
    models[fold]['counter'].fit(np.hstack([train[str(col)].loc[train_index] for col in config.text_cols]))
    train_data = np.hstack([models[fold]['counter'].transform(train[str(col)].loc[train_index]) for col in config.text_cols])
    test_data = np.hstack([models[fold]['counter'].transform(train[str(col)].loc[test_index]) for col in config.text_cols])

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
    hold_out_test_data = np.hstack([models[fold]['counter'].transform(test[str(col)]) for col in config.text_cols])
    prediction['rank'] += - models[fold]['model'].predict(hold_out_test_data) / nfolds
prediction = prediction.sort_values(by=['context_id', 'rank'])
prediction[['context_id', 'reply_id']].to_csv(path+'/sub_svd_counter_all_tfidf_lgbm.tsv',header=None, index=False, sep=' ')
prediction.to_csv(path+'/sub_rank_svd_counter_all_tfidf_lgbm.tsv', sep=' ')

pickle.dump(models, open(path+'/all_tfidf_svd_lgbm.pickle.dat', 'wb'))
logging.info('submission saved!')