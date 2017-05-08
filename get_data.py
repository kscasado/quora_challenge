import os
import re
import numpy as np
import pandas as pd
from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

########################################
## set directories and parameters
########################################
BASE_DIR = 'data/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300


act = 'relu'

import spacy
nlp = spacy.load('en')
print('Processing text dataset')

#cleans text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    rx = re.compile('\W+')
    res = rx.sub(' ',string).strip()
    res = res.replace('\'','')
    if remove_stopwords and stem_words:
        return ' '.join([x.lemma_ for x in nlp(res) if not x.is_stop ])
    if remove_stopwords and not stem_words
        return ' '.join([str(x) for x in nlp(res) if not x.is_stop])
    if stem_words and not remove_stopwords:
        return ' '.join([x.lemma_ for x in nlp(res)])
    else
        return ' '.join([str(x) for x in nlp(res) if not x.is_stop])

df = pd.read_csv(TRAIN_DATA_FILE, encoding='utf-8')
## If this is taking long using nlp.pipe would increase efficiency with multithreading
texts_1 = df['question1'].astype('str').apply(lambda x: text_to_wordlist(unicode(x)))
texts_2 = df['question2'].astype('str').apply(lambda x: text_to_wordlist(unicode(x)))
labels = df['is_duplicate']

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))
data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)
########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')
nb_words = min(MAX_NB_WORDS, len(word_index))+1
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
## Uses spacy built in word vectors
for word, i in word_index.items():
    embedding_matrix[i] = nlp(unicode(word)).vector
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
pickle.dump(data_1 ,open('data/q1_train_2.p','wb'))
pickle.dump(data_2,open('data/q2_train_2.p','wb'))
pickle.dump(test_data_1,open('data/q1_test_2.p','wb'))
pickle.dump(test_data_2,open('data/q2_test_2.p','wb'))
pickle.dump(embedding_matrix,open('data/embedd_matrix_2.p','wb'))
