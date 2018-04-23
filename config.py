import numpy as np

path = '/home/jupyter/ML_track'
#path = '/home/ML_track'

#features_path = path + '/data/svd_cv_features_svd_200'
features_path = path + '/data/tfidf_cv_features'

prep_mode = None
ranking_mode = False
nfolds = 10
text_cols = [1, 2, 3, 5]
svd_n_components = 200
max_features = 200000
num_round = 5
lower_is_better = True
early_stopping = False

params = {
    'n_jobs': -1, 
    'n_estimators': 150, 
    'max_depth': -1,
    'learning_rate': 0.03, 
    'min_data_in_leaf': 20,
    #'lambda_l2': 0.15,
    'subsample_for_bin':300000,
    'max_bin':1024,
    #'boosting_type':'dart',
    'num_leaves': 50,
    #'verbose': 0,

    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': 1,
    'feature_fraction': 0.9,
    'feature_fraction_seed': 1,

    #'application': 'lambdarank', 
    #'max_position': 20,
    #'label_gain':[0,1,2,3],
    #'metric':'ndcg',

    'objective': 'regression_l2',
    'random_state': 42
}

opt = {
    'n_estimators': range(50, 250, 50),
    'num_leaves': range(30, 100, 10),
    'learning_rate': np.linspace(0.01, 0.2, num=10),
    'min_data_in_leaf': range(10, 50, 10)
}