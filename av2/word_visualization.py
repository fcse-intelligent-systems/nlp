import pandas as pd
from nltk import word_tokenize
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_data():
    return pd.read_csv('../data/train.En.csv', usecols=['tweet', 'sarcastic']).dropna()


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

    model = Word2Vec(sentences, size=50, min_count=15, window=5, sg=1)
    words = model.wv.index2word
    vectors = model.wv.vectors
    save('word2vecSG.iSarcasmEval.50d', words, vectors)

    tsne = TSNE(n_components=2, random_state=0)
    vectors_2d = tsne.fit_transform(vectors)

    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
    for label, x, y in zip(words, vectors_2d[:, 0], vectors_2d[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
