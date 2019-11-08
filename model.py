"""
This script defines ML/NN models for text classification.
"""

import tensorflow as tf

def cnn_1d(n_input, n_vocab, **params):
    """
    Return a keras model with 1D Convolution. The rough idea is:
        Input Sequence (with length of n_input)
        |
        Word Embedding
        |
        1D Convolution
        |
        Full Connection
        |
        FC to 1 output (binary)
    
    The original keras example is here:
    https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py
    """

    embedding_size = 50
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(n_vocab,
                                  embedding_size,
                                  input_length=n_input), # specify this argument,
                                        # so that model.summary() can be more explicit
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(filters,
                               kernel_size,
                               padding='valid',
                               activation='relu',
                               strides=1),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(hidden_dims, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    return model


def lstm_1layer(n_input, n_vocab):
    """
    Return a keras model with 1 layer of LSTM cells. The rough idea is:
        Input Sequence (with length of n_input)
        |
        Word Embedding
        |
        RNN cells (LSTM)
        |
        FC to 1 output (binary)
    
    The original keras example is here:
    https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
    """
    embedding_size = 128
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(n_vocab,
                                  embedding_size,
                                  input_length=n_input),
        tf.keras.layers.LSTM(units=embedding_size,
                             dropout=0.2,
                             recurrent_dropout=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def cnn_lstm(n_input, n_vocab):
    """
    Return a keras model with 1 layer CNN and then 1 layer LSTM:
        Input Sequence (length: n_input)
        |
        Word Embedding
        |
        1D Convolution
        |
        RNN cells (LSTM)
        |
        FC to 1 output
    The original keras example is here:
    https://github.com/keras-team/keras/blob/master/examples/imdb_cnn_lstm.py
    """
    embedding_size = 128
    kernel_size = 5
    n_filters = 64
    pool_size = 4
    lstm_output_size = 70

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(n_vocab,
                                  embedding_size,
                                  input_length=n_input),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv1D(n_filters,
                               kernel_size,
                               padding='valid',
                               activation='relu',
                               strides=1),
        tf.keras.layers.MaxPooling1D(pool_size),
        tf.keras.layers.LSTM(lstm_output_size),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    return model


def nn_1hid(n_input, **params):
    '''
    Return a keras model of vanilla neural network with 1 hidden layer.

    Used after document level vectorization.
    '''

    # n_input = param['n_input']
    n_hidden = 256
    dropout = 0.25
    n_output = 1
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(n_hidden, input_shape=(n_input,), activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(n_output, activation='sigmoid'),
    ])

    return model


from sklearn.base import BaseEstimator, TransformerMixin
class TextConverter(BaseEstimator, TransformerMixin):
    '''
    Connect each element in a dataset example to produce a string.
    This will fit in the sklearn's built-in tf-idf vectorizer.
    '''
    def __init__(self):
        pass
        # self.feature_names = feature_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X:'Tuple list') -> 'string list':
        str_list = []
        for seq in X:
            seq = [str(val) for val in seq]
            str_list.append(' '.join(seq))
        return str_list


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
def pipeline_tfidf():
    '''
    Return a sklearn pipeline for tf-idf preprocessing.
    '''
    return Pipeline([
        ('text_converter', TextConverter()),
        ('tfidf_vectorizer', TfidfVectorizer())
    ])
