import pickle
import logging
import os
from functools import partial
from itertools import combinations
from shutil import copyfile

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import ClassifierMixin, clone

import config
from scripts import *

path = config.path +'/data/'+os.path.basename(__file__)
os.mkdir(path)
copyfile(config.path+'/config.py', path+'/config.py')

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(path + '/train.log', mode='w')])

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
lgbm = LGBMRegressor(n_estimators=500, n_jobs=-1)
cv = KFold(n_splits=nfolds, shuffle=True, random_state=42)

fold, cv_score = 0, 0
models = {i:{'counter':counters_pipe, 'model':{get_model_(key, params=params) for key, params in config.blend.items()}, 'pred':{}, 'loss':{}, 'ndcg':{}} for i in range(nfolds)}

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
    models[fold]['ndcg']['lr'] = make_prediction(train.loc[test_index], test_data, models[fold]['model']['lr'])

    message = 'LEARNER (TRAINING): LINEAR MODEL fold no: %s' % fold + ' cross val. score: %s (ndcg: %s)' % (models[fold]['loss']['lr'], models[fold]['ndcg']['lr'])
    logging.info(message)

    fit_params = {
        'eval_metric': ['mse'],
        'eval_set': [(train_data, train['target'].loc[train_index]),
                     (test_data, train['target'].loc[test_index])],
        'early_stopping_rounds': config.num_round
    }
    models[fold]['model']['xgb'].fit(train_data, train['target'].loc[train_index],
                           early_stopping_rounds=fit_params['early_stopping_rounds'],
                           eval_metric=fit_params['eval_metric'], eval_set=fit_params['eval_set'], verbose=False)
    models[fold]['pred']['xgb'] = models[fold]['model']['xgb'].predict(test_data)
    models[fold]['loss']['xgb'] = mean_squared_error(train['target'].loc[test_index], models[fold]['pred']['xgb'])

    message = 'LEARNER (TRAINING): XGB fold no: %s' % fold + ' cross val. score: %s' % models[fold]['loss']['xgb']
    logging.info(message)

    n_iterations = models[fold]['model']['rf'].n_estimators
    early = EarlyStopping(models[fold]['model']['rf'],
                          max_n_estimators=n_iterations,
                          scorer=partial(es_scorer, test_data, train['target'].loc[test_index]),
                          n_min_iterations=config.num_round,
                          scale=1)
    early.fit(train_data, train['target'].loc[train_index])

    models[fold]['model']['rf'] = early.estimator_
    models[fold]['pred']['rf'] = models[fold]['model']['rf'].predict(test_data)
    models[fold]['loss']['rf'] = mean_squared_error(train['target'].loc[test_index], models[fold]['pred']['rf'])

    message = 'LEARNER (TRAINING): RF fold no: %s' % fold + ' cross val. score: %s' % models[fold]['loss']['rf']
    logging.info(message)

    models[fold]['model']['cat'].fit(train_data, train['target'].loc[train_index], eval_set=(test_data, train['target'].loc[test_index]))
    models[fold]['pred']['cat'] = models[fold]['model']['cat'].predict(test_data)
    models[fold]['loss']['cat'] = mean_squared_error(train['target'].loc[test_index], models[fold]['pred']['cat'])

    message = 'LEARNER (TRAINING): CATBOOST fold no: %s' % fold + ' cross val. score: %s' % models[fold]['loss']['cat']
    logging.info(message)

    models[fold]['model']['lgbm'].fit(train_data, train['target'].loc[train_index],
                            eval_set=[(test_data, train['target'].loc[test_index])],
                            eval_metric='mse', verbose=False,
                            early_stopping_rounds=config.num_round)
    models[fold]['pred']['lgbm'] = models[fold]['model']['lgbm'].predict(test_data, num_iteration=models[fold]['model']['lgbm'].best_iteration_)
    models[fold]['loss']['lgbm'] = mean_squared_error(train['target'].loc[test_index], models[fold]['pred']['lgbm'])

    message = 'LEARNER (TRAINING): LIGHT GBM fold no: %s' % fold + ' cross val. score: %s' % models[fold]['loss']['lgbm']
    logging.info(message)

    #add blend

    #ndcg score
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
    #add blending
    for model in models[fold]['model']:
        prediction['rank'] += - models[fold]['model'][model].predict(hold_out_test_data) / nfolds
prediction = prediction.sort_values(by=['context_id', 'rank'])
prediction[['context_id', 'reply_id']].to_csv(path+'/sub_blend_svd.tsv',header=None, index=False, sep=' ')
prediction.to_csv(path+'/sub_rank_blend_svd.tsv', sep=' ')

pickle.dump(models, open(path+'/blend_svd.pickle.dat', 'wb'))
logging.info('submission saved!')

        #averaged solutions
        tmp['clfr']['averaged'] = {
            'lr': tmp['clfr']['lr'],
            'xgb': tmp['clfr']['xgb'],
            'rf': tmp['clfr']['rf'],
            'cat': tmp['clfr']['cat'],
            'lgbm': tmp['clfr']['lgbm']
        }

        combs = [item for sublist in [[list(j) for j in combinations(tmp['clfr']['averaged'].keys(), i)] for i in range(2,len(tmp['clfr']['averaged'].keys()))] for item in sublist]

        average = {
             '_'.join(model_names):
                {
                    'pred': {
                        'mean': sum([tmp['pred'][model_name] for model_name in model_names]) / len(model_names),
                        'geom': np.power(np.multiply.reduce([tmp['pred'][model_name] for model_name in model_names]), (1./len(model_names))),
                        'harmonic': len(model_names) / sum([1./tmp['pred'][model_name] for model_name in model_names])
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
                if way == 'geom':
                    if config.lower_is_better:
                        average[key]['loss'][way] = np.inf
                    else:
                        average[key]['loss'][way] = -np.inf
                    continue
                try:
                    average[key]['loss'][way] = self.scorer(y_test_, pred)
                except:
                    average[key]['loss'][way] = self.scorer(y_test_.values, pred)

        if not config.lower_is_better:
            best_combination = max([max([(key, value) for way, value in average[key]['loss'].items()], key=operator.itemgetter(1)) for key in average.keys()], key=operator.itemgetter(1))[0]
            averaging_mode = max([(key, value) for key, value in average[best_combination]['loss'].items()], key=operator.itemgetter(1))[0]
        else:
            best_combination = min([min([(key, value) for way, value in average[key]['loss'].items()], key=operator.itemgetter(1)) for key in average.keys()], key=operator.itemgetter(1))[0]
            averaging_mode = min([(key, value) for key, value in average[best_combination]['loss'].items()], key=operator.itemgetter(1))[0]

        tmp['pred']['averaged'] = average[best_combination]['pred'][averaging_mode]
        tmp['loss']['averaged'] = average[best_combination]['loss'][averaging_mode]

        #search for the best solution
        if not config.lower_is_better:
            self.best_model = max([(key, value) for key, value in tmp['loss'].items()], key=operator.itemgetter(1))[0]
        else:
            self.best_model = min([(key, value) for key, value in tmp['loss'].items()], key=operator.itemgetter(1))[0]

        self.models[it][fold]['best'] = self.best_model
        if self.best_model == 'averaged':
            self.models[it][fold]['mode'] = averaging_mode
            self.models[it][fold]['clfr'] = {key:value for key, value in tmp['clfr'][self.best_model].items() if key in best_combination.split('_')}
            message = 'LEARNER (TRAINING): fold no: %s' % fold + ' best model: %s (%s)' % (self.best_model, averaging_mode) + ' best combination: %s' % ' '.join(best_combination.split('_')) + ' loss: %s' % tmp['loss'][self.best_model] + '\n'
        else:
            self.models[it][fold]['clfr'] = {self.best_model: tmp['clfr'][self.best_model]}
            message = 'LEARNER (TRAINING): fold no: %s' % fold + ' best model: %s' % self.best_model + ' loss: %s' % tmp['loss'][self.best_model] + '\n'

        print(message)
        self.logs.append(message)

        self.prediction_loss += tmp['loss'][self.best_model] / self.nfolds