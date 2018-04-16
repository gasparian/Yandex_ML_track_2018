path = '/home/ML_track'

text_cols = [1, 2, 3, 5]
svd_n_components = 100
max_features = 200000
num_round = 50
lower_is_better = True

blend = {
	'lr': None,
	'xgb': {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 5, 'nthread': -1}, 
	'rf': {'n_estimators': 300, 'max_depth': 5, 'n_jobs': -1}, 
	'lgbm': {'n_jobs': -1, 'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03, 'num_leaves': 40, 'max_depth': 5}, 
	'cat': {      
		'random_seed': 42,
		'od_type': 'Iter',
		'od_wait': num_round,
		'iterations': 500, 
		'learning_rate': 0.03, 
		'depth': 6, 
		'use_best_model': True,
		'verbose': False,
		'l2_leaf_reg': 3,
		'rsm': 0.5,
		'thread_count': 4
	}
}