import pickle
import logging
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import pipeline
from sklearn.model_selection import KFold
from scipy.sparse import hstack

import config
from scripts import *

path = config.path +'/data/tfidf_cv_features'
try:
    os.mkdir(path)
except:
    pass

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(path + '/prep_cv_train_data.log', mode='w'), logging.StreamHandler()])

train = pd.read_csv(config.path+'/data/train_modified.tsv', sep=' ', index_col=0)
test = pd.read_csv(config.path+'/data/test_modified.tsv', sep=' ', index_col=0)

train.fillna('', inplace=True)
test.fillna('', inplace=True)

counters_pipe = pipeline.FeatureUnion(
    n_jobs = -1,
    transformer_list = [
    ('chars_features', TfidfVectorizer(
            analyzer=u'char', ngram_range=(2, 5), tokenizer=None,
            max_features=config.max_features, strip_accents=None, max_df=0.9, min_df=2, lowercase=False)),
    ('words_features', TfidfVectorizer(
       analyzer=u'word', ngram_range=(1, 3), tokenizer=None, use_idf=True,
       max_features=config.max_features, strip_accents=None, max_df=0.9, min_df=2, lowercase=False)),
])

cv = KFold(n_splits=config.nfolds, shuffle=True, random_state=42)
splits = list(cv.split(train))

fold = 0
models = {i:counters_pipe for i in range(config.nfolds)}

for train_index, test_index in splits:
    logging.info('Make features: fold#%i...' % fold)
    models[fold].fit(np.hstack([train[str(col)].loc[train_index] for col in config.text_cols]))
    train_data = hstack([models[fold].transform(train[str(col)].loc[train_index]) for col in config.text_cols])
    test_data = hstack([models[fold].transform(train[str(col)].loc[test_index]) for col in config.text_cols])
    hold_out_test_data = hstack([models[fold].transform(test[str(col)]) for col in config.text_cols])

    logging.info(train_data.shape)
    logging.info(test_data.shape)
    
    pickle.dump(train_data, open(path+'/train_fold_%i.pickle.dat' % fold, 'wb'))
    pickle.dump(test_data, open(path+'/test_fold_%i.pickle.dat' % fold, 'wb'))
    pickle.dump(hold_out_test_data, open(path+'/hold_test_fold_%i.pickle.dat' % fold, 'wb'))

    logging.info('fold#%i saved!' % fold)
    fold += 1

pickle.dump(models, open(path+'/all_tfidf_svd.pickle.dat', 'wb'))
pickle.dump(splits, open(path+'/all_tfidf_svd_splits.pickle.dat', 'wb'))
logging.info('TFIDF preparation finished!')