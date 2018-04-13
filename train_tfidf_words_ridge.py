import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.model_selection import KFold
import scipy.sparse

import config
from scripts import *


train = pd.read_csv(config.path+'/data/train_modified.tsv', sep=' ', index_col=0)
test = pd.read_csv(config.path+'/data/test_modified.tsv', sep=' ', index_col=0)

train.fillna('', inplace=True)
test.fillna('', inplace=True)

tfidf = TfidfVectorizer(
       analyzer=u'word', ngram_range=(1, 3), tokenizer=None, use_idf=True,
       max_features=config.max_features, strip_accents=None, max_df=0.9, min_df=2, lowercase=False)

nfolds = round(len(train) / len(test))
lr = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15)
cv = KFold(n_splits=nfolds, shuffle=True, random_state=42)

fold, cv_score = 0, 0
models = {i:{'tfidf':tfidf, 'model':lr} for i in range(nfolds)}

for train_index, test_index in cv.split(train):
    print('make features...')
    models[fold]['tfidf'].fit(np.hstack([train[str(col)].loc[train_index] for col in config.text_cols]))
    train_data = scipy.sparse.hstack([models[fold]['tfidf'].transform(train[str(col)].loc[train_index]) for col in config.text_cols]).tocsr()
    test_data = scipy.sparse.hstack([models[fold]['tfidf'].transform(train[str(col)].loc[test_index]) for col in config.text_cols]).tocsr()

    print(train_data.shape)
    print(test_data.shape)
    
    print('fitting model...')
    models[fold]['model'].fit(train_data, train['target'].loc[train_index])

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
    hold_out_test_data = scipy.sparse.hstack([models[fold]['tfidf'].transform(test[str(col)]) for col in config.text_cols]).tocsr()
    prediction['rank'] += - models[fold]['model'].predict(hold_out_test_data) / nfolds
prediction = prediction.sort_values(by=['context_id', 'rank'])
prediction[['context_id', 'reply_id']].to_csv(config.path+'/data/sub_words_tfidf_ridge.tsv',header=None, index=False, sep=' ')
prediction.to_csv(config.path+'/data/sub_rank_words_tfidf_ridge.tsv', sep=' ')

pickle.dump(models, open(config.path+'/models/words_tfidf_ridge.pickle.dat', 'wb'))
print('submission saved!')