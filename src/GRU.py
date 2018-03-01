# coding: utf-8
"""
Convolutional NN to classify govuk content to level2 taxons

From https://www.kaggle.com/maupson/pooled-gru-fasttext
"""

import json
import os
import warnings
import logging
import logging.config
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, TensorBoard
from utils import RocAucEvaluation, get_coefs

warnings.filterwarnings('ignore')
np.random.seed(42)

#os.environ['OMP_NUM_THREADS'] = '4'

# Load environmental variables

DATADIR=os.environ.get('DATADIR')
LOGGING_CONFIG = 'logging.conf'

logging.config.fileConfig(LOGGING_CONFIG)
logger = logging.getLogger('GRU')

# Set data locations

EMBEDDING_FILE = os.path.join(DATADIR, 'crawl-300d-2M.vec')
TRAIN_DATA = os.path.join(DATADIR, 'train.csv')
TEST_DATA = os.path.join(DATADIR, 'test.csv')
SUBMISSION_DATA = os.path.join(DATADIR, 'sample_submission.csv')
EMBEDDINGS_INDEX_FILE = os.path.join(DATADIR, 'embedding_index.json')

# Set hyperparameters

NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300

BATCH_SIZE = 256
EPOCHS = 2
TRAIN_SIZE=0.95


logger.info('---- Model hyperparameters ----')
logger.info('MAX_SEQUENCE_LENGTH: %s', MAX_SEQUENCE_LENGTH)
logger.info('NUM_WORDS:           %s', NUM_WORDS)
logger.info('EMBEDDING_DIM:       %s', EMBEDDING_DIM)
logger.info('NUM_WORDS:           %s', NUM_WORDS)
logger.info('EPOCHS:              %s', EPOCHS)
logger.info('BATCH_SIZE:          %s', BATCH_SIZE)
logger.info('TRAIN_SIZE:          %s', TRAIN_SIZE)
logger.info('---- Other parameters ----')
logger.info('LOGGING_CONFIG:     %s', LOGGING_CONFIG)
logger.info('DATADIR:            %s', DATADIR)
logger.info('--------------------------')

# Load data

logger.info('Loading data files')

train = pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)
submission = pd.read_csv(SUBMISSION_DATA)

# Fill NAs

logger.info('Filling NAs in training and test data')

X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values

# tokenizer

logger.info('Instantiating tokenizer')
tokenizer = text.Tokenizer(num_words=NUM_WORDS)

logger.info('Fitting tokenizer')
tokenizer.fit_on_texts(list(X_train) + list(X_test))

logger.info('Converting texts to squences')

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

logger.info('Padding sequences to %s words', MAX_SEQUENCE_LENGTH)

x_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
x_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

logger.info('Checking for existing embeddings file')

if os.path.exists('EMBEDDINGS_INDEX_FILE'):
    logger.info('%s exists, loading from json', EMBEDDINGS_INDEX_FILE)
    with open(EMBEDDINGS_INDEX_FILE, 'rb') as f:
        embeddings_index = json.load(f)
else:
    logger.info('%s does not exist, creating', EMBEDDINGS_INDEX_FILE)
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
    logger.info('Writing embedding index to %s', EMBEDDINGS_INDEX_FILE)
    with open(EMBEDDINGS_INDEX_FILE, 'wb') as f:
        json.dump(embeddings_index, f)

word_index = tokenizer.word_index
nb_words = min(NUM_WORDS, len(word_index))

logger.info('Building embedding matrix')

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= NUM_WORDS: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

logger.info('Defining model')

def get_model():
    inp = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    x = Embedding(NUM_WORDS, EMBEDDING_DIM, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model = get_model()

logger.info('Creating train/test split with train/test split: %s/%s',
            TRAIN_SIZE, 1-TRAIN_SIZE)

X_tra, X_val, y_tra, y_val = train_test_split(
    x_train, y_train, train_size=TRAIN_SIZE, random_state=233
)

RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

tb = TensorBoard(
    log_dir='./tb_logs', histogram_freq=1,
    write_graph=True, write_images=False
    )

logger.info('Fitting model')

hist = model.fit(X_tra, y_tra, batch_size=BATCH_SIZE, epochs=EPOCHS, \
                 validation_data=(X_val, y_val), callbacks=[RocAuc, tb], verbose=1)


y_pred = model.predict(x_test, batch_size=BATCH_SIZE)

submission[["toxic", "severe_toxic", "obscene", 
            "threat", "insult", "identity_hate"]] = y_pred

submission.to_csv('submission.csv', index=False)
