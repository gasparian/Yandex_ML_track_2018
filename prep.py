import logging
from collections import OrderedDict
import operator
import pickle

import pandas as pd
import numpy as np
import nltk
from gensim.utils import deaccent
nltk.download('wordnet')

import config

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(config.path + '/data/prep.log', mode='w'), TqdmLoggingHandler()])

logging.info('reading data...')
train = pd.read_csv(config.path+'/data/train.tsv', sep='\t', quoting=3, header=None)
test = pd.read_csv(config.path+'/data/public.tsv', sep='\t', quoting=3, error_bad_lines=False, header=None)

skipgram_ru = KeyedVectors.load_word2vec_format(config.path+'/models/ruwikiruscorpora_upos_skipgram_300_2_2018.vec.gz')
lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
vocab = {}
for word in tqdm(skipgram_ru.vocab.keys(), desc='skipgram', ascii=True):
    new_word = lmtzr.lemmatize(deaccent(word.split('_')[0]))
    new_skipgram_ru[new_word] = skipgram_ru[word]
del skipgram_ru

train.fillna('', inplace=True)
test.fillna('', inplace=True)

# расставляем значения по формуле хорошесть * confidence
# Good (3)
# Neutral (2)
# Bad (1)
def rank2num(st):
    if st == 'good':
        return 3
    else:
        if st == 'neutral':
            return 2
        else:
            return 1

train['rank'] = train[6].apply(rank2num)
train['target'] = train['rank'] * train[7]

class getTokens:
    
    def __init__(self, garbage=[]):
        self.tokenizer = nltk.RegexpTokenizer('\w+')        
        self.lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
        self.garbage = set(garbage)
    
    def transform(self, text):
        text = self.tokenizer.tokenize(text)
        text = [self.lmtzr.lemmatize(deaccent(word.lower())) for word in text]
        text = [word.lower() for word in text if len(word) > 1 and word.lower() not in self.garbage]
        return ' '.join(text)

logging.info('transforimng data...')
tokenize = getTokens()
for col in config.text_cols:
    train[col] = train[col].apply(tokenize.transform)
    test[col] = test[col].apply(tokenize.transform)

train.to_csv(config.path+'/data/train_modified.tsv', sep=' ')
test.to_csv(config.path+'/data/test_modified.tsv', sep=' ')

logging.info('Data saved!')