import pandas as pd
import numpy as np
from nltk import word_tokenize
from scripts.word_embeddings import load_embedding_weights
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy


def load_data():
	return pd.read_csv('../data/train.En.csv', usecols=['tweet', 'sarcastic']).dropna()


def tokenize(data):
	data['tweet_tokens'] = data['tweet'].apply(lambda x: word_tokenize(x.lower()))


def create_vocabulary(sentence_tokens):
	vocabulary = set()
	for tokens in sentence_tokens:
		vocabulary.update(tokens)

	vocabulary = list(vocabulary)
	word_to_id = {word: index for word, index in zip(vocabulary, range(len(vocabulary)))}
	return vocabulary, word_to_id


if __name__ == '__main__':
	dataset = load_data()

	tokenize(dataset)

	sentences = dataset['tweet_tokens'].values
	labels = dataset['sarcastic'].values

	vocabulary, word_to_id = create_vocabulary(sentences)

	embeddings = load_embedding_weights(vocabulary, 50, 'glove')

	dataset['tweet_indices'] = dataset['tweet_tokens'].apply(lambda x: np.array([word_to_id[i] for i in x]))
	sentence_indices = dataset['tweet_indices'].values
	padded_sentences = pad_sequences(sentence_indices, 10)

	x_train, x_test, y_train, y_test = train_test_split(padded_sentences, labels, test_size=0.1,
														random_state=0, stratify=labels)

	model = Sequential()
	model.add(Embedding(input_dim=len(vocabulary), output_dim=50,
						weights=[embeddings], trainable=False))

	# model.add(LSTM(128, return_sequences=True))
	# model.add(LSTM(128))

	model.add(GRU(128))

	# model.add(Bidirectional(LSTM(128)))

	model.add(Dense(1, activation='sigmoid'))

	model.compile(optimizer=Adam(learning_rate=0.01),
				  loss=binary_crossentropy,
				  metrics=['accuracy'])

	model.fit(x_train, y_train, epochs=15, batch_size=32, verbose=2)

	model.evaluate(x_test, y_test)
