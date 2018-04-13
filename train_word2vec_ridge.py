import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import pipeline
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from gensim.models import KeyedVectors

import config
from scripts import *


train = pd.read_csv(config.path+'/data/train_modified.tsv', sep=' ', index_col=0)
test = pd.read_csv(config.path+'/data/test_modified.tsv', sep=' ', index_col=0)

train.fillna('', inplace=True)
test.fillna('', inplace=True)

skipgram_ru = KeyedVectors.load_word2vec_format(config.path+'/models/ruwikiruscorpora_upos_skipgram_300_2_2018.vec.gz')
lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
new_skipgram_ru = {}
for word in skipgram_ru.vocab.keys():
    new_word = lmtzr.lemmatize(deaccent(word.split('_')[0]))
    new_skipgram_ru[new_word] = skipgram_ru[word]
del skipgram_ru

tfidf = TfidfVectorizer(
       analyzer=u'word', ngram_range=(1, 3), tokenizer=None, use_idf=True,
       max_features=config.max_features, strip_accents=None, max_df=0.9, min_df=2, lowercase=False)

def weighter(transformed, tfidf, new_skipgram_ru):
    data = None
    for i in range(transformed.shape[0]):
        features = 0
        for j, word in enumerate(tfidf.vocabulary_):
            if word in new_skipgram_ru.keys():
                features += new_skipgram_ru[word] * transformed[i, tfidf.vocabulary_[word]]
        if data is None:
            data = features
        else:
            data = np.vstack([data, features])
    return data

nfolds = round(len(train) / len(test))
lgbm = LGBMRegressor(n_estimators=500, n_jobs=-1)
cv = KFold(n_splits=nfolds, shuffle=True, random_state=42)

fold, cv_score = 0, 0
models = {i:{'tfidf':tfidf, 'model':lgbm} for i in range(nfolds)}

for train_index, test_index in cv.split(train):
    print('make features...')
    models[fold]['tfidf'].fit(np.hstack([train[str(col)].loc[train_index] for col in config.text_cols]))
    train_data = np.hstack([weighter(models[fold]['tfidf'].transform(train[str(col)].loc[train_index]), models[fold]['tfidf'], new_skipgram_ru) for col in config.text_cols])
    test_data = np.hstack([weighter(models[fold]['tfidf'].transform(train[str(col)].loc[test_index]), models[fold]['tfidf'], new_skipgram_ru) for col in config.text_cols])

    print(train_data.shape)
    print(test_data.shape)
    
    print('fitting model...')
    models[fold]['model'].fit(train_data, train['target'].loc[train_index],
                        eval_set=[(test_data, train['target'].loc[test_index])],
                        eval_metric='mae', verbose=False,
                        early_stopping_rounds=config.num_round)

    #self.models_params['lgbm']['params']['n_estimators'] = estimators['clfr'][key].best_iteration_

    print('make prediction...')
    prediction = make_prediction(train.loc[test_index], test_data, models[fold]['model'])
    
    score = 0
    uniques = prediction['context_id'].unique()
    for Id in uniques:
        tmp = prediction[prediction['context_id'] == Id]['reply_id'].values.tolist()
        score += ndcg_at_k(tmp, len(set(tmp))) / len(uniques)
    score *= 100000
    cv_score += score / nfolds
    
    print('fold#%i: ndcg = %i' % (fold, int(score)))
    fold += 1
print('averaged cv score: ndcg = %i' % (int(cv_score)))

prediction = pd.DataFrame()
prediction['context_id'] = test['0']
prediction['reply_id'] = test['4']
prediction['rank'] = 0
for fold in range(nfolds):
    hold_out_test_data = np.hstack([weighter(models[fold]['tfidf'].transform(test[str(col)]), models[fold]['tfidf'], new_skipgram_ru) for col in config.text_cols])
    prediction['rank'] += - models[fold]['model'].predict(hold_out_test_data) / nfolds
prediction = prediction.sort_values(by=['context_id', 'rank'])
prediction[['context_id', 'reply_id']].to_csv(config.path+'/data/sub_w2v_lgbm.tsv',header=None, index=False, sep=' ')
prediction.to_csv(config.path+'/data/sub_rank_w2v_lgbm.tsv', sep=' ')

pickle.dump(models, open(config.path+'/models/w2v_lgbm.pickle.dat', 'wb'))
print('submission saved!')