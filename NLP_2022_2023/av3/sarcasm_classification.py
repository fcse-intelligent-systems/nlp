import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, TweetTokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
from scripts.word_embeddings import load_embedding_weights


def load_data():
	return pd.read_csv('../data/train.En.csv', usecols=['tweet', 'sarcastic']).dropna()


def tokenize(data):
	data['tweet_tokens'] = data['tweet'].apply(lambda x: word_tokenize(x.lower()))


def create_vocabulary(sentences):
	vocabulary = set()
	for sentence in sentences:
		vocabulary.update(sentence)

	vocabulary = list(vocabulary)
	word_to_id = {word: index for word, index in zip(vocabulary, range(len(vocabulary)))}

	return vocabulary, word_to_id


if __name__ == '__main__':
	data = load_data()

	tokenize(data)

	tweets = data['tweet_tokens'].values
	labels = data['sarcastic'].values

	vocabulary, word_to_id = create_vocabulary(tweets)

	data['tweet_indices'] = data['tweet_tokens'].apply(lambda x: np.array([word_to_id[word] for word in x]))

	tweet_indices = data['tweet_indices'].values
	padded_indices = pad_sequences(tweet_indices, 15)

	x_train, x_test, y_train, y_test = train_test_split(padded_indices, labels, test_size=0.1,
														random_state=0, stratify=labels)

	embeddings = load_embedding_weights(vocabulary, 50, 'glove')

	model = Sequential()
	model.add(Embedding(input_dim=len(vocabulary), output_dim=50, weights=[embeddings], trainable=False))
	model.add(LSTM(32, return_sequences=False))
	# model.add(LSTM(16))
	model.add(Dense(1, activation='sigmoid'))

	print(model.summary())

	# model.add(Dense(3, activation='softmax'))

	model.compile(optimizer=Adam(learning_rate=0.01),
				  loss=binary_crossentropy,
				  metrics=['accuracy'])

	model.fit(x_train, y_train, epochs=5, batch_size=16, verbose=2)

	print()
