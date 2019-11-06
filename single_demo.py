from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import cnn_1d, lstm_1layer, cnn_lstm

# Parameters
n_vocab = 20000     # vocabulary size
maxlen = 100        # length of each example (paragraph)
models = {          # model selection
    1:cnn_1d,
    2:lstm_1layer,
    3:cnn_lstm,
}
n = 3              # number of model
batch_size = 30    # some models are batch-size sensitive
epochs = 3

# Load imdb dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=n_vocab,
    # skip_top=0,
    # maxlen=None,
    # seed=113,
    # start_char=1,
    # oov_char=2,
    # index_from=3,
)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# Model set-up
model = models[n](n_vocab, maxlen)
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training
model.fit(x_train, y_train,
          batch_size=batch_size,
          verbose=1,
          epochs=epochs,
          validation_data=(x_test, y_test))