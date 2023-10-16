import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec


def load_data():
    return pd.read_csv('../data/train.En.csv', usecols=['tweet']).dropna()


def tokenize(data):
    data['tweet_tokens'] = data['tweet'].apply(lambda x: word_tokenize(x.lower()))


def save(name, words, vectors):
    with open(f'../data/{name}.txt', 'w+', encoding='utf-8') as doc:
        for word, vector in zip(words, vectors):
            doc.write(word + ' ' + ' '.join(str(value) for value in vector))
            doc.write('\n')


if __name__ == '__main__':
    dataset = load_data()
    tokenize(dataset)

    sentences = dataset['tweet_tokens'].values

    word2vec = Word2Vec(sentences, vector_size=50, min_count=15,
                        window=3, sg=1)

    vectors = word2vec.wv.vectors
    id_to_word = word2vec.wv.index_to_key

    save('word2vec', id_to_word, vectors)
