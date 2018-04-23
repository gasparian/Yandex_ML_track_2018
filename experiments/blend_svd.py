import pickle
import logging
import os
import operator
from functools import partial
from itertools import combinations
from shutil import copyfile

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import config
from scripts import *

path = config.path +'/data/'+os.path.basename(__file__).split('.')[0]
try:
    os.mkdir(path)
except:
    pass
copyfile(config.path+'/config.py', path+'/config.py')

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(path + '/train.log', mode='w'), logging.StreamHandler()])

logging.info('read data...')
train = pd.read_csv(config.path+'/data/train_modified.tsv', sep=' ', index_col=0)
test = pd.read_csv(config.path+'/data/test_modified.tsv', sep=' ', index_col=0)

train.fillna('', inplace=True)
test.fillna('', inplace=True)

counters_pipe = pipeline.FeatureUnion(
    n_jobs = -1,
    transformer_list = [
    ('chars_features', pipeline.Pipeline([
        ('chars_counter', CountVectorizer(
            analyzer=u'char', ngram_range=(2, 4), tokenizer=None,
            max_features=config.max_features, strip_accents=None, max_df=0.9, min_df=2, lowercase=False)),
        ('chars_tsvd', TruncatedSVD(n_components=config.svd_n_components, n_iter=25, random_state=42))])),
    ('words_features', pipeline.Pipeline([
        ('words_counter', TfidfVectorizer(
       analyzer=u'word', ngram_range=(1, 3), tokenizer=None, use_idf=True,
       max_features=config.max_features, strip_accents=None, max_df=0.9, min_df=2, lowercase=False)),
        ('words_tsvd', TruncatedSVD(n_components=config.svd_n_components, n_iter=25, random_state=42))])),
])

nfolds = round(len(train) / len(test))
cv = KFold(n_splits=nfolds, shuffle=True, random_state=42)

fold, cv_loss = 0, 0
models = {i:{'counter':counters_pipe, 'model':{key:get_model_(key, params=params) for key, params in config.blend.items()}, 'pred':{}, 'loss':{}, 'ndcg':{}} for i in range(nfolds)}

for train_index, test_index in cv.split(train):
    logging.info('make features...')
    models[fold]['counter'].fit(np.hstack([train[str(col)].loc[train_index] for col in config.text_cols]))
    train_data = np.hstack([models[fold]['counter'].transform(train[str(col)].loc[train_index]) for col in config.text_cols])
    test_data = np.hstack([models[fold]['counter'].transform(train[str(col)].loc[test_index]) for col in config.text_cols])

    logging.info(train_data.shape)
    logging.info(test_data.shape)
    logging.info('fitting model...')

    models[fold]['model']['lr'].fit(train_data, train['target'].loc[train_index])
    models[fold]['pred']['lr'] = models[fold]['model']['lr'].predict(test_data)
    models[fold]['loss']['lr'] = mean_squared_error(train['target'].loc[test_index], models[fold]['pred']['lr'])
    models[fold]['ndcg']['lr'] = make_prediction(train.loc[test_index], test_data, models[fold]['model']['lr'], loss=True)

    message = 'LEARNER (TRAINING): LINEAR MODEL fold no: %s' % fold + ' cross val. score: %s (ndcg: %s)' % (models[fold]['loss']['lr'], models[fold]['ndcg']['lr'])
    logging.info(message)

    if config.early_stopping:
        fit_params = {
            'eval_metric': 'rmse',
            'eval_set': [(train_data, train['target'].loc[train_index]),
                         (test_data, train['target'].loc[test_index])],
            'early_stopping_rounds': config.num_round
        }
        models[fold]['model']['xgb'].fit(train_data, train['target'].loc[train_index],
                               early_stopping_rounds=fit_params['early_stopping_rounds'],
                               eval_metric=fit_params['eval_metric'], eval_set=fit_params['eval_set'], verbose=True)
    else:
        models[fold]['model']['xgb'].fit(train_data, train['target'].loc[train_index])

    models[fold]['pred']['xgb'] = models[fold]['model']['xgb'].predict(test_data)
    models[fold]['loss']['xgb'] = mean_squared_error(train['target'].loc[test_index], models[fold]['pred']['xgb'])
    models[fold]['ndcg']['xgb'] = make_prediction(train.loc[test_index], test_data, models[fold]['model']['xgb'], loss=True)

    message = 'LEARNER (TRAINING): XGB fold no: %s' % fold + ' cross val. score: %s (ndcg: %s)' % (models[fold]['loss']['xgb'], models[fold]['ndcg']['xgb'])
    logging.info(message)

    if config.early_stopping:
        models[fold]['model']['cat'].fit(train_data, train['target'].loc[train_index], eval_set=(test_data, train['target'].loc[test_index]))
    else:
        models[fold]['model']['cat'].fit(train_data, train['target'].loc[train_index])

    models[fold]['pred']['cat'] = models[fold]['model']['cat'].predict(test_data)
    models[fold]['loss']['cat'] = mean_squared_error(train['target'].loc[test_index], models[fold]['pred']['cat'])
    models[fold]['ndcg']['cat'] = make_prediction(train.loc[test_index], test_data, models[fold]['model']['cat'], loss=True)

    message = 'LEARNER (TRAINING): CATBOOST fold no: %s' % fold + ' cross val. score: %s (ndcg: %s)' % (models[fold]['loss']['cat'], models[fold]['ndcg']['cat'])
    logging.info(message)

    if config.early_stopping:
        models[fold]['model']['lgbm'].fit(train_data, train['target'].loc[train_index],
                                eval_set=[(test_data, train['target'].loc[test_index])],
                                eval_metric='l2', verbose=True,
                                early_stopping_rounds=config.num_round)
        models[fold]['pred']['lgbm'] = models[fold]['model']['lgbm'].predict(test_data, num_iteration=models[fold]['model']['lgbm'].best_iteration_)
    else:
        models[fold]['model']['lgbm'].fit(train_data, train['target'].loc[train_index])
        models[fold]['pred']['lgbm'] = models[fold]['model']['lgbm'].predict(test_data)

    models[fold]['loss']['lgbm'] = mean_squared_error(train['target'].loc[test_index], models[fold]['pred']['lgbm'])
    models[fold]['ndcg']['lgbm'] = make_prediction(train.loc[test_index], test_data, models[fold]['model']['lgbm'], loss=True)

    message = 'LEARNER (TRAINING): LIGHT GBM fold no: %s' % fold + ' cross val. score: %s (ndcg: %s)' % (models[fold]['loss']['lgbm'], models[fold]['ndcg']['lgbm'])
    logging.info(message)

    #add blend
    models[fold]['model']['averaged'] = {
        'lr': models[fold]['model']['lr'],
        'xgb': models[fold]['model']['xgb'],
        'cat': models[fold]['model']['cat'],
        'lgbm': models[fold]['model']['lgbm']
    }

    combs = [item for sublist in [[list(j) for j in combinations(models[fold]['model']['averaged'].keys(), i)] for i in range(2,len(models[fold]['model']['averaged'].keys()))] for item in sublist]

    average = {
         '_'.join(model_names):
            {
                'pred': {
                    'mean': sum([models[fold]['pred'][model_name] for model_name in model_names]) / len(model_names),
                    'geom': np.power(np.abs(np.multiply.reduce([models[fold]['pred'][model_name] for model_name in model_names])), (1./len(model_names))),
                    'harmonic': len(model_names) / sum([1./models[fold]['pred'][model_name] for model_name in model_names])
                },
                'loss': {
                    'mean': None,
                    'geom': None,
                    'harmonic': None
                }
            }
        for model_names in combs
    }

    for key in average.keys():
        for way, pred in average[key]['pred'].items():
            average[key]['loss'][way] = mean_squared_error(train['target'].loc[test_index], pred)

    if not config.lower_is_better:
        best_combination = max([max([(key, value) for way, value in average[key]['loss'].items()], key=operator.itemgetter(1)) for key in average.keys()], key=operator.itemgetter(1))[0]
        averaging_mode = max([(key, value) for key, value in average[best_combination]['loss'].items()], key=operator.itemgetter(1))[0]
    else:
        best_combination = min([min([(key, value) for way, value in average[key]['loss'].items()], key=operator.itemgetter(1)) for key in average.keys()], key=operator.itemgetter(1))[0]
        averaging_mode = min([(key, value) for key, value in average[best_combination]['loss'].items()], key=operator.itemgetter(1))[0]

    models[fold]['pred']['averaged'] = average[best_combination]['pred'][averaging_mode]
    models[fold]['loss']['averaged'] = average[best_combination]['loss'][averaging_mode]

    #search for the best solution
    if not config.lower_is_better:
        best_model = max([(key, value) for key, value in models[fold]['loss'].items()], key=operator.itemgetter(1))[0]
    else:
        best_model = min([(key, value) for key, value in models[fold]['loss'].items()], key=operator.itemgetter(1))[0]

    models[fold]['best'] = best_model
    models[fold]['mode'] = None
    if best_model == 'averaged':
        models[fold]['mode'] = averaging_mode
        models[fold]['model'] = {key:value for key, value in models[fold]['model'][best_model].items() if key in best_combination.split('_')}
        message = 'LEARNER (TRAINING): fold no: %s' % fold + ' best model: %s (%s)' % (best_model, averaging_mode) + ' best combination: %s' % ' '.join(best_combination.split('_')) + ' loss: %s' % models[fold]['loss'][best_model] + '\n'
    else:
        models[fold]['model'] = models[fold]['model'][best_model]
        message = 'LEARNER (TRAINING): fold no: %s' % fold + ' best model: %s' % best_model + ' loss: %s' % models[fold]['loss'][best_model] + '\n'
    logging.info(message)

    cv_loss += models[fold]['loss'][best_model] / nfolds
    fold += 1

logging.info('averaged cv loss: loss = %s' % (cv_loss))

pickle.dump(models, open(path+'/blend_svd.pickle.dat', 'wb'))
logging.info('Models saved!')

prediction = pd.DataFrame()
prediction['context_id'] = test['0']
prediction['reply_id'] = test['4']
prediction['rank'] = 0

logging.info('Predicting on hold out test...')
for fold in range(nfolds):
    hold_out_test_data = np.hstack([models[fold]['counter'].transform(test[str(col)]) for col in config.text_cols])
    if models[fold]['best'] == 'averaged':
        if models[fold]['mode'] == 'mean':
            scores = np.ones((hold_out_test_data.shape[0]))
            for key in models[fold]['model'].keys():
                scores += models[fold]['model'][key].predict(hold_out_test_data) / len(models[fold]['model'])
        elif models[fold]['mode'] == 'geom':
            scores = np.ones((hold_out_test_data.shape[0]))
            for key in models[fold]['model'].keys():
                scores *= models[fold]['model'][key].predict(hold_out_test_data)
            scores = np.power(np.abs(scores), (1/len(models[fold]['model'])))
        elif models[fold]['mode'] == 'harmonic':
            scores = np.zeros((hold_out_test_data.shape[0]))
            for key in models[fold]['model'].keys():
                    scores += 1./models[fold]['model'][key].predict(hold_out_test_data)
            scores = len(models[fold]['model'])/scores
    else:
        scores = models[fold]['model'].predict(hold_out_test_data)
    prediction['rank'] += - scores / nfolds

logging.info('Hold out test - predicted!')
prediction = prediction.sort_values(by=['context_id', 'rank'])
prediction[['context_id', 'reply_id']].to_csv(path+'/sub_blend_svd.tsv',header=None, index=False, sep=' ')
prediction.to_csv(path+'/sub_rank_blend_svd.tsv', sep=' ')
logging.info('Submission saved!')