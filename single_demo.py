# Load imdb dataset
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

n_vocab = 5000     # vocabulary size
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=n_vocab,
    # skip_top=0,
    # maxlen=None,
    # seed=113,
    # start_char=1,
    # oov_char=2,
    # index_from=3,
)

maxlen = 400    # length of each example
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# Model set-up
from model import cnn_1d, lstm_1layer
models = {
    1:cnn_1d,
    2:lstm_1layer,
}
n = 1     # model selection
model = models[n](n_vocab, maxlen)
model.summary()

# Training
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 512
epochs = 3
model.fit(x_train, y_train,
          batch_size=batch_size,
          verbose=2,
          epochs=epochs,
          validation_data=(x_test, y_test))