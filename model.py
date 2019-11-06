"""
This script defines ML/NN models for text classification.
"""

import tensorflow as tf

def cnn_1d(vocab_size, seq_len):
    """
    Return a keras model with 1D Convolution. The rough idea is:
        Input Sequence (with length of seq_len)
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
        tf.keras.layers.Embedding(vocab_size,
                                  embedding_size,
                                  input_length=seq_len), # specify this argument,
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


def lstm_1layer(vocab_size, seq_len):
    """
    Return a keras model with 1 layer of LSTM cells. The rough idea is:
        Input Sequence (with length of seq_len)
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
        tf.keras.layers.Embedding(vocab_size,
                                  embedding_size,
                                  input_length=seq_len),
        tf.keras.layers.LSTM(units=embedding_size,
                             dropout=0.2,
                             recurrent_dropout=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def cnn_lstm(vocab_size, seq_len):
    """
    Return a keras model with 1 layer CNN and then 1 layer LSTM:
        Input Sequence (length: seq_len)
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
        tf.keras.layers.Embedding(vocab_size,
                                  embedding_size,
                                  input_length=seq_len),
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