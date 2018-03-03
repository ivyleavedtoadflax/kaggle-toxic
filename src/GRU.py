# coding: utf-8
"""
RNN

From https://www.kaggle.com/maupson/pooled-gru-fasttext
"""

import logging
import logging.config
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import (Input, Dense, Embedding, SpatialDropout1D, concatenate, 
    GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, 
    BatchNormalization, CuDNNGRU)
#from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, TensorBoard
from datetime import datetime

LOGGING_CONFIG = 'logging.conf'
logging.config.fileConfig(LOGGING_CONFIG)
logger = logging.getLogger('GRU')

"""
Helper functions for model evaluation
"""

def get_coefs(word, *arr): 
    """
    """   
    return word, np.asarray(arr, dtype='float32')
    
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            logging.info("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

warnings.filterwarnings('ignore')
RANDOM_SEED=42
np.random.seed(RANDOM_SEED)

#os.environ['OMP_NUM_THREADS'] = '4'

# Load environmental variables

DATADIR = os.environ.get('DATADIR') # = '../input/jigsaw-toxic-comment-classification-challenge/'
EMBEDDING_DIR = os.environ.get('EMBEDDING_DIR') # = '../input/fasttext-crawl-300d-2m/'

# Set data locations

EMBEDDING_FILE = os.path.join(EMBEDDING_DIR, 'crawl-300d-2M.vec')
TRAIN_DATA = os.path.join(DATADIR, 'train.csv')
TEST_DATA = os.path.join(DATADIR, 'test.csv')
SUBMISSION_DATA = os.path.join(DATADIR, 'sample_submission.csv')
EMBEDDINGS_INDEX_FILE = os.path.join(DATADIR, 'embeddings_index.json')
TB_LOG_DIR = datetime.now().strftime('%Y%m%d-%H%M%S')
# Set hyperparameters

NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300
TRAIN_BATCH_SIZE = 1024
PREDICTION_BATCH_SIZE = 1024
EPOCHS = 5
TRAIN_SIZE = 0.95
DROPOUT_1 = 0.2
LEARNING_RATE = 0.001
# https://research.fb.com/wp-content/uploads/2017/06/imagenet1kin1h5.pdf?
LEARNING_RATE_DECAY = 0.0

logging.info('------- Model hyperparameters -------')
logging.info('MAX_SEQUENCE_LENGTH:   %s', MAX_SEQUENCE_LENGTH)
logging.info('NUM_WORDS:             %s', NUM_WORDS)
logging.info('EMBEDDING_DIM:         %s', EMBEDDING_DIM)
logging.info('NUM_WORDS:             %s', NUM_WORDS)
logging.info('EPOCHS:                %s', EPOCHS)
logging.info('TRAIN_BATCH_SIZE:      %s', TRAIN_BATCH_SIZE)
logging.info('PREDICTION_BATCH_SIZE: %s', PREDICTION_BATCH_SIZE)
logging.info('TRAIN_SIZE:            %s', TRAIN_SIZE)
logging.info('DROPOUT_1:             %s', DROPOUT_1)
logging.info('LEARNING_RATE:         %s', LEARNING_RATE)
logging.info('LEARNING_RATE_DECAY:   %s', LEARNING_RATE_DECAY)
logging.info('------- Other parameters -------')
logging.info('RANDOM_SEED:           %s', RANDOM_SEED)
logging.info('DATADIR:               %s', DATADIR)
logging.info('EMBEDDING_DIR:         %s', EMBEDDING_DIR)
logging.info('TB_LOG_DIR:            %s', TB_LOG_DIR)
logging.info('--------------------------------')

# Load data

logging.info('Loading data files')

train = pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)
submission = pd.read_csv(SUBMISSION_DATA)

# Fill NAs

logging.info('Filling NAs in training and test data')

X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values

# tokenizer

logging.info('Instantiating tokenizer')
tokenizer = text.Tokenizer(num_words=NUM_WORDS)

logging.info('Fitting tokenizer')
tokenizer.fit_on_texts(list(X_train) + list(X_test))

logging.info('Converting texts to squences')

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

logging.info('Padding sequences')

x_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
x_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

logging.info('Creating embeddings_index')

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
    
word_index = tokenizer.word_index
nb_words = min(NUM_WORDS, len(word_index))

logging.info('Building embedding matrix')

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= NUM_WORDS: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

logging.info('Defining model')

logging.info('Customise adam optimiser with learning rate decay')

#adam_ = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE_DECAY)

tb = TensorBoard(
    log_dir=TB_LOG_DIR, histogram_freq=0,
    write_graph=True, write_images=False
    )


def get_model():
    inp = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    x = Embedding(NUM_WORDS, EMBEDDING_DIM, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(DROPOUT_1)(x)
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

logging.info(model.summary())

logging.info('Creating train/test split with train/test split')

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, \
    train_size=TRAIN_SIZE, random_state=RANDOM_SEED)

RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

logging.info('Fitting model')

hist = model.fit(X_tra, y_tra, batch_size=TRAIN_BATCH_SIZE, epochs=EPOCHS, \
    validation_data=(X_val, y_val), callbacks=[RocAuc, tb], verbose=1)
                 
logging.info(hist)


logging.info('Running prediction')

y_pred = model.predict(x_test, batch_size=PREDICTION_BATCH_SIZE)

submission[["toxic", "severe_toxic", "obscene", 
            "threat", "insult", "identity_hate"]] = y_pred

submission.to_csv('submission.csv', index=False)
