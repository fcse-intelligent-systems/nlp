import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec


def load_data():
	return pd.read_csv('../data/train.En.csv', usecols=['tweet']).dropna()


def tokenize(data):
	data['tweet_tokens'] = data['tweet'].apply(lambda x: word_tokenize(x.lower()))


def learn_word2vec(sentences):
	return Word2Vec(sentences, vector_size=50, min_count=15, window=5, sg=1)


def save(name, words, vectors):
	with open(f'../data/{name}.txt', 'w+', encoding='utf-8') as doc:
		for word, vector in zip(words, vectors):
			doc.write(word+' '+' '.join(str(value) for value in vector))
			doc.write('\n')


if __name__ == '__main__':
	data = load_data()
	tokenize(data)

	sentences = data['tweet_tokens'].values

	model = learn_word2vec(sentences)

	words = model.wv.index_to_key
	vectors = model.wv.vectors

	save('Word2VecSG', words, vectors)
