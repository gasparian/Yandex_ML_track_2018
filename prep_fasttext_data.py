import pandas as pd
import numpy as np
import pickle
import re
from tqdm import tqdm
import config
import logging

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(config.path + '/prep_fasttext.log', mode='w'), logging.StreamHandler()])

train_length = len(pd.read_csv(config.path+'/data/train_modified.tsv', sep=' ', index_col=0))
test_length = len(pd.read_csv(config.path+'/data/test_modified.tsv', sep=' ', index_col=0))

for col in config.text_cols:
    logging.info('Converting train col#%i...' % col)
    df = np.empty((train_length, 300))
    with open(config.path+'/data/train_text_%i_vectors.txt' % col, 'r') as f:
        i = 0
        for line in tqdm(f, ascii=True, desc='train; column#%i' % col):
            splitted = line.split()
            try:
                array = np.array(splitted).astype(float)[np.newaxis, :]
                if array.shape[1] > 300:
                    array = array[:, array.shape[1]-300:]
            except:
                for j, w in enumerate(splitted):
                    try:
                        int(w)
                    except:
                        try:
                            float(w)
                            break
                        except:
                            pass
                array = np.array(splitted[j:]).astype(float)[np.newaxis, :]
            df[i, :] = array
            i += 1
    pickle.dump(df, open(config.path+'/data/train_text_%i_vectors.pickle.dat' % col, 'wb'))

    logging.info('Converting test col#%i...' % col)
    df = np.empty((test_length, 300))
    with open(config.path+'/data/test_text_%i_vectors.txt' % col, 'r') as f:
        i = 0
        for line in tqdm(f, ascii=True, desc='test; column#%i' % col):
            splitted = line.split()
            try:
                array = np.array(splitted).astype(float)[np.newaxis, :]
                if array.shape[1] > 300:
                    array = array[:, array.shape[1]-300:]
            except:
                for j, w in enumerate(splitted):
                    try:
                        int(w)
                    except:
                        try:
                            float(w)
                            break
                        except:
                            pass
                array = np.array(splitted[j:]).astype(float)[np.newaxis, :]
            df[i, :] = array
            i += 1
    pickle.dump(df, open(config.path+'/data/test_text_%i_vectors.pickle.dat' % col, 'wb'))
logging.info('Finished!')