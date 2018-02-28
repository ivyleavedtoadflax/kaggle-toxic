"""
From https://www.kaggle.com/maupson/pooled-gru-fasttext
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback


warnings.filterwarnings('ignore')
np.random.seed(42)

os.environ['OMP_NUM_THREADS'] = '4'

# Load environmental variables

DATADIR=os.environ.get('DATADIR')

# Set data locations

EMBEDDING_FILE = os.path.join(DATADIR, 'crawl-300d-2M.vec')
TRAIN_DATA = os.path.join(DATADIR, 'train.csv')
TEST_DATA = os.path.join(DATADIR, 'test.csv')
SUBMISSION_DATA = os.path.join(DATADIR, 'sample_submission.csv')

# Load tokenizer and padded train and test from pickle

TOKENIZER = os.path.join(DATADIR, 'tokenizer.pkl')
X_TRAIN = os.path.join(DATADIR, 'x_train.pkl')
X_TEST = os.path.join(DATADIR, 'x_test.pkl')
EMBEDDING_MATRIX = os.path.join(DATADIR, 'embedding_matrix.pkl')

# Load data

train = pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)
submission = pd.read_csv(SUBMISSION_DATA)

# Fill NAs

X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values


max_features = 30000
maxlen = 100
embed_size = 300

batch_size = 32
epochs = 2

# Check whether these data already exist, if so load from pickle

# tokenizer

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))

X_train = tokenizer.texts_to_sequences(X_train)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = tokenizer.texts_to_sequences(X_test)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

print("defining model")

def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
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



print("Splitting dataset")

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

print("Fitting model")

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[RocAuc], verbose=2)


y_pred = model.predict(x_test, batch_size=1024)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('submission.csv', index=False)
