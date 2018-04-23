import logging
from collections import OrderedDict
import operator
import pickle
import re

import pandas as pd
import numpy as np
import nltk
from gensim.utils import deaccent
nltk.download('wordnet')

import config
from scripts import *

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(config.path + '/prep.log', mode='w'), logging.StreamHandler()])

logging.info('reading data...')
train = pd.read_csv(config.path+'/data/train.tsv', sep='\t', quoting=3, header=None)
test = pd.read_csv(config.path+'/data/final.tsv', sep='\t', quoting=3, error_bad_lines=False, header=None)

train.fillna('', inplace=True)
test.fillna('', inplace=True)

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
    
    def __init__(self, garbage=[], mode='gensim'):
        self.tokenizer = nltk.RegexpTokenizer('\w+')        
        self.lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
        self.mode = mode
        self.garbage = set(garbage)
    
    def transform(self, text):
        if self.mode == 'gensim':
            text = self.tokenizer.tokenize(text)
            text = [self.lmtzr.lemmatize(deaccent(word.lower())) for word in text]
            text = [word.lower() for word in text if len(word) > 1 and word.lower() not in self.garbage]
            text = ' '.join(text)
        else:
            text = text.lower().strip()
            # Isolate punctuation
            text = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', text)
            # # Remove some special characters
            # text = re.sub(r'([\;\:\|•«\n])', ' ', text)
            # Replace numbers and symbols with language
            text = text.replace('&', ' и ')
            text = text.replace('@', ' в ')
            text = text.replace('0', ' ноль ')
            text = text.replace('1', ' один ')
            text = text.replace('2', ' два ')
            text = text.replace('3', ' три ')
            text = text.replace('4', ' четыре ')
            text = text.replace('5', ' пять ')
            text = text.replace('6', ' шесть ')
            text = text.replace('7', ' семь ')
            text = text.replace('8', ' восемь ')
            text = text.replace('9', ' девять ')
            text = ' '.join(re.findall('\w+', text))
        return text

logging.info('transforimng data...')
tokenize = getTokens(mode=config.prep_mode)
for col in config.text_cols:
    train[col] = train[col].apply(tokenize.transform)
    with open(config.path+'/data/train_text_%i.txt' % col, 'w') as f:
        for row in train[col]:
            f.write(row+'\n')

    test[col] = test[col].apply(tokenize.transform)
    with open(config.path+'/data/test_text_%i.txt' % col, 'w') as f:
        for row in test[col]:
            f.write(row+'\n')

train.to_csv(config.path+'/data/train_modified.tsv', sep=' ')
test.to_csv(config.path+'/data/test_modified.tsv', sep=' ')

logging.info('Data saved!')