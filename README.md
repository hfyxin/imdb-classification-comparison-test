# Working on it...
# IMDB Sentiment Classification Comparison Test

IMDB movie review dataset is a popular dataset used for text classification. Several methods will be implemented to tackle the problem and their results will be compared:
- TF-IDF and Logistic Regression
- TF-IDF and MLP
- Word-Embedding and LSTM
- Word-Embedding and CNN
- Word-Embedding and CNN + RNN

With vocabulary size of 4000, sequence length trimmed to 200, some initial results (accuracy on test data) are:
- TF-IDF + LR: 
- TF-IDF + FC: ~89%, but tend to overfit.
- LSTM 1 layer: 83% ~ 86%, very slow.
- CNN + FC: ~88%
- CNN + RNN: ~88%
- bi-directional RNN: 

(No fine-tune yet)

(TBD)

