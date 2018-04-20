path = '/home/ML_track'

text_cols = [1, 2, 3, 5]
svd_n_components = 200
max_features = 250000
num_round = 5
lower_is_better = True
early_stopping = False

blend = {
	'lr': {'loss':'squared_loss', 'penalty':'l2', 'alpha':0.0001, 'l1_ratio':0.15, 'max_iter':1000, 'tol':1e-3},
	'xgb': {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 5, 'n_jobs': 4, 'tree_method':'hist'},
	'lgbm': {
		'n_jobs': -1, 
		'n_estimators': 500, 
		'max_depth': 5, 
		'learning_rate': 0.03, 
		'num_leaves': 40, 
		'max_depth': 5, 
		'random_state': 42
	}, 
	'cat': {      
		'loss_function':'RMSE',
		'random_seed': 42,
		'iterations': 500, 
		'learning_rate': 0.03, 
		'depth': 6, 
		'verbose': True,
		'l2_leaf_reg': 3,
		'rsm': 0.5,
		'thread_count': 4
	}
}

if early_stopping:
	blend['cat']['custom_loss'] = ['RMSE']
	blend['cat']['od_type'] = 'Iter'
	blend['cat']['od_wait'] = num_round
	blend['cat']['use_best_model'] = True