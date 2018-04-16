import pickle
import logging
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import pipeline
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from gensim.models import KeyedVectors
import nltk
nltk.download('wordnet')
from gensim.utils import deaccent
from tqdm import tqdm

import config
from scripts import *

path = config.path +'/data/'+os.path.basename(__file__)
os.mkdir(path)
from shutil import copyfile
copyfile(config.path+'/config.py', path+'/config.py')

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(path + '/train.log', mode='w'), TqdmLoggingHandler()])

train = pd.read_csv(config.path+'/data/train_modified.tsv', sep=' ', index_col=0)
test = pd.read_csv(config.path+'/data/test_modified.tsv', sep=' ', index_col=0)

train.fillna('', inplace=True)
test.fillna('', inplace=True)

skipgram_ru = KeyedVectors.load_word2vec_format(config.path+'/models/ruwikiruscorpora_upos_skipgram_300_2_2018.vec.gz')
lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
new_skipgram_ru = {}
for word in tqdm(skipgram_ru.vocab.keys(), desc='skipgram', ascii=True):
    new_word = lmtzr.lemmatize(deaccent(word.split('_')[0]))
    new_skipgram_ru[new_word] = skipgram_ru[word]
del skipgram_ru
new_skipgram_ru = pd.DataFrame.from_dict(new_skipgram_ru, orient='index')

tfidf = TfidfVectorizer(
       analyzer=u'word', ngram_range=(1, 3), tokenizer=None, use_idf=True,
       max_features=config.max_features, strip_accents=None, max_df=0.9, min_df=2, lowercase=False)

nfolds = round(len(train) / len(test))
lgbm = LGBMRegressor(n_estimators=500, n_jobs=-1)
cv = KFold(n_splits=nfolds, shuffle=True, random_state=42)

fold, cv_score = 0, 0
models = {i:{'tfidf':tfidf, 'model':lgbm} for i in range(nfolds)}

for train_index, test_index in cv.split(train):
    logging.info('make features...')
    models[fold]['tfidf'].fit(np.hstack([train[str(col)].loc[train_index] for col in config.text_cols]))
    logging.info('feature extractor fitted!')
    train_data = np.hstack([weighter(models[fold]['tfidf'].transform(train[str(col)].loc[train_index]), models[fold]['tfidf'], new_skipgram_ru, col) for col in config.text_cols])
    test_data = np.hstack([weighter(models[fold]['tfidf'].transform(train[str(col)].loc[test_index]), models[fold]['tfidf'], new_skipgram_ru, col) for col in config.text_cols])
    logging.info('w2v ready!')

    logging.info(train_data.shape)
    logging.info(test_data.shape)
    
    logging.info('fitting model...')
    models[fold]['model'].fit(train_data, train['target'].loc[train_index],
                        eval_set=[(test_data, train['target'].loc[test_index])],
                        eval_metric='mse', verbose=False,
                        early_stopping_rounds=config.num_round)

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
    hold_out_test_data = np.hstack([weighter(models[fold]['tfidf'].transform(test[str(col)]), models[fold]['tfidf'], new_skipgram_ru, col) for col in config.text_cols])
    prediction['rank'] += - models[fold]['model'].predict(hold_out_test_data) / nfolds
prediction = prediction.sort_values(by=['context_id', 'rank'])
prediction[['context_id', 'reply_id']].to_csv(path+'/sub_w2v_lgbm.tsv',header=None, index=False, sep=' ')
prediction.to_csv(path+'/sub_rank_w2v_lgbm.tsv', sep=' ')

pickle.dump(models, open(path+'/w2v_lgbm.pickle.dat', 'wb'))
logging.info('submission saved!')