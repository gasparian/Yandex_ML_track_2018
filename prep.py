import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')

from collections import OrderedDict
from gensim.utils import deaccent
import operator
import pickle

import config

print('reading data...')
train = pd.read_csv(config.path+'/data/train.tsv', sep='\t', quoting=3, header=None)
test = pd.read_csv(config.path+'/data/public.tsv', sep='\t', quoting=3, error_bad_lines=False, header=None)

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
        text = [word for word in text if len(word) > 1 and word not in self.garbage]
        return ' '.join(text)

print('transforimng data...')
tokenize = getTokens()
for col in config.text_cols:
    train[col] = train[col].apply(tokenize.transform)
    test[col] = test[col].apply(tokenize.transform)

train.to_csv(config.path+'/data/train_modified.tsv', sep=' ')
test.to_csv(config.path+'/data/test_modified.tsv', sep=' ')

print('Data saved!')