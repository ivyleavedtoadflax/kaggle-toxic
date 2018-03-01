# coding: utf-8
"""
Helper functions for model evaluation
"""

import tensorflow as tf
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score)



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
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

class Metrics(Callback):
    """
    """

    def __init__(self, logger):
        self.logger = logger
        self.dev_f1s = []
        self.dev_recalls = []
        self.dev_precisions = []

    def on_train_begin(self, logs={}):
        self.dev_f1s = []
        self.dev_recalls = []
        self.dev_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        dev_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        dev_targ = self.model.validation_data[1]

        self.dev_f1s.append(f1_score(dev_targ, dev_predict, average='micro'))
        self.dev_recalls.append(recall_score(dev_targ, dev_predict))
        self.dev_precisions.append(precision_score(dev_targ, dev_predict))

        f1 = f1_score(dev_targ, dev_predict, average='micro')
        precision = precision_score(dev_targ, dev_predict),
        recall = recall_score(dev_targ, dev_predict)

        self.logger.info("Metrics: - dev_f1: %s — dev_precision: %s — dev_recall %s", f1, precision, recall)
        return

def f1(y_true, y_pred):
    """
    Use Recall and precision metrics to calculate harmonic mean (f1)

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1 = 2*((precision*recall)/(precision+recall))

    return f1
