# coding: utf-8
"""
RNN

From https://www.kaggle.com/maupson/pooled-gru-fasttext
"""

import logging
import logging.config
import os
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import (Input, Dense, Embedding, concatenate, GRU,
                          Bidirectional, GlobalAveragePooling1D,
                          GlobalMaxPooling1D, BatchNormalization)
from keras.preprocessing import text, sequence
from keras.callbacks import TensorBoard
from utils import RocAucEvaluation, get_coefs

PROJECT_ROOT = os.environ.get('PROJECT_ROOT')

# Setup logging

LOGGING_CONFIG = os.path.join(PROJECT_ROOT, 'logging.conf')
logging.config.fileConfig(LOGGING_CONFIG)
logger = logging.getLogger('GRU')

warnings.filterwarnings('ignore')
RANDOM_SEED = int(os.environ.get('RANDOM_SEED'))
np.random.seed(RANDOM_SEED)

#os.environ['OMP_NUM_THREADS'] = '4'

DATADIR = os.environ.get('DATADIR') # = '../input/jigsaw-toxic-comment-classification-challenge/'
EMBEDDING_DIR = os.environ.get('EMBEDDING_DIR') # = '../input/fasttext-crawl-300d-2m/'

# Set data locations

EMBEDDING_FILE = os.path.join(EMBEDDING_DIR, 'crawl-300d-2M.vec')
TRAIN_DATA = os.path.join(DATADIR, 'train.csv')
TEST_DATA = os.path.join(DATADIR, 'test.csv')
SUBMISSION_DATA = os.path.join(DATADIR, 'sample_submission.csv')
EMBEDDINGS_INDEX_FILE = os.path.join(DATADIR, 'embeddings_index.json')
NOW = datetime.now().strftime('%Y%m%d-%H%M%S')
TB_LOG_DIR = os.path.join(PROJECT_ROOT, 'tb_logs', NOW)
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Add logging filehandler

filehandler = logging.FileHandler(os.path.join(LOG_DIR, NOW + '.log'))
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
filehandler.setFormatter(formatter)
logger.addHandler(filehandler)

# Set hyperparameters

NUM_WORDS = int(os.environ.get('NUM_WORDS'))
MAX_SEQUENCE_LENGTH = int(os.environ.get('MAX_SEQUENCE_LENGTH'))
EMBEDDING_DIM = int(os.environ.get('EMBEDDING_DIM'))
TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE'))
PREDICTION_BATCH_SIZE = int(os.environ.get('PREDICTION_BATCH_SIZE'))
EPOCHS = int(os.environ.get('EPOCHS'))
TRAIN_SIZE = float(os.environ.get('TRAIN_SIZE'))
DROPOUT_1 = float(os.environ.get('DROPOUT_1'))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE'))
# https://research.fb.com/wp-content/uploads/2017/06/imagenet1kin1h5.pdf?
LEARNING_RATE_DECAY = float(os.environ.get('LEARNING_RATE_DECAY'))

logger.info('------- Model hyperparameters -------')
logger.info('MAX_SEQUENCE_LENGTH:   %s', MAX_SEQUENCE_LENGTH)
logger.info('NUM_WORDS:             %s', NUM_WORDS)
logger.info('EMBEDDING_DIM:         %s', EMBEDDING_DIM)
logger.info('EPOCHS:                %s', EPOCHS)
logger.info('TRAIN_BATCH_SIZE:      %s', TRAIN_BATCH_SIZE)
logger.info('PREDICTION_BATCH_SIZE: %s', PREDICTION_BATCH_SIZE)
logger.info('TRAIN_SIZE:            %s', TRAIN_SIZE)
logger.info('DROPOUT_1:             %s', DROPOUT_1)
logger.info('LEARNING_RATE:         %s', LEARNING_RATE)
logger.info('LEARNING_RATE_DECAY:   %s', LEARNING_RATE_DECAY)
logger.info('------- Other parameters -------')
logger.info('RANDOM_SEED:           %s', RANDOM_SEED)
logger.info('DATADIR:               %s', DATADIR)
logger.info('EMBEDDING_DIR:         %s', EMBEDDING_DIR)
logger.info('TB_LOG_DIR:            %s', TB_LOG_DIR)
logger.info('LOG_DIR:               %s', LOG_DIR)
logger.info('--------------------------------')

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

logger.info('Padding sequences')

x_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
x_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

logger.info('Creating embeddings_index')

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
    
word_index = tokenizer.word_index
nb_words = min(NUM_WORDS, len(word_index))

logger.info('Building embedding matrix')

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= NUM_WORDS: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

logger.info('Defining model')

tb = TensorBoard(
    log_dir=TB_LOG_DIR, histogram_freq=0,
    write_graph=True, write_images=False
    )


def get_model():
    inp = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    x = Embedding(NUM_WORDS, EMBEDDING_DIM, weights=[embedding_matrix])(inp)
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

logger.info(model.summary())

logger.info('Creating train/test split with train/test split')

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, \
    train_size=TRAIN_SIZE, random_state=RANDOM_SEED)

RocAuc = RocAucEvaluation(logger=logger, validation_data=(X_val, y_val), interval=1)

logger.info('Fitting model')

hist = model.fit(X_tra, y_tra, batch_size=TRAIN_BATCH_SIZE, epochs=EPOCHS, \
    validation_data=(X_val, y_val), callbacks=[RocAuc, tb], verbose=1)
                 
logger.info('Running prediction')

y_pred = model.predict(x_test, batch_size=PREDICTION_BATCH_SIZE)

submission[["toxic", "severe_toxic", "obscene", 
            "threat", "insult", "identity_hate"]] = y_pred

submission.to_csv('submission.csv', index=False)
