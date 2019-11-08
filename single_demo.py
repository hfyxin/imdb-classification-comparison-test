from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import cnn_1d, lstm_1layer, cnn_lstm, nn_1hid, pipeline_tfidf

# Parameters
n_vocab = 4000     # vocabulary size
maxlen = 200        # length of each example, only for word-embedding
models = {          # model selection
    1:cnn_1d,
    2:lstm_1layer,
    3:cnn_lstm,
    4:nn_1hid,
}
n = 3              # number of model
preproc = 'word-embedding'  # word-embedding / tf-idf
batch_size = 32    # some models are batch-size sensitive
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

# preprocessing for word embedding
if preproc == 'word-embedding':
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
# preprocessing for tf-idf
elif preproc == 'tf-idf':
    vectorizer = pipeline_tfidf()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
else:
    print('Error', preproc)
    
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# Model set-up
n_input = x_train.shape[1]
model = models[n](n_input, n_vocab)
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