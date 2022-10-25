import pandas as pd
from nltk import word_tokenize
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_data():
    return pd.read_csv('../data/train.En.csv', usecols=['tweet', 'sarcastic']).dropna()


def tokenize(data):
    data['tweet_tokens'] = data['tweet'].apply(lambda x: word_tokenize(x.lower()))


if __name__ == '__main__':
    dataset = load_data()

    tokenize(dataset)

    sentences = dataset['tweet_tokens'].values

    model = Word2Vec(sentences, size=50, min_count=1, window=5, sg=0)

    dataset['embeddings'] = dataset['tweet_tokens'].apply(lambda x: model[x])
    dataset['avg_embeddings'] = dataset['embeddings'].apply(lambda x: x.mean(axis=0))

    vectors = dataset['avg_embeddings'].values.tolist()
    labels = dataset['sarcastic'].values.tolist()

    tsne = TSNE(n_components=2, random_state=0)
    vectors_2d = tsne.fit_transform(vectors)

    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=labels, cmap='jet', alpha=0.7)
    plt.show()
