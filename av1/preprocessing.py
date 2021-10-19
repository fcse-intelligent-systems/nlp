import pandas as pd
from nltk import word_tokenize, pos_tag, WordNetLemmatizer
from nltk.stem import PorterStemmer


def load_data():
    return pd.read_csv('../data/train.En.csv', index_col=[0])


def tokenize(sentence):
    return word_tokenize(sentence)


def find_pos_tag(sentence_tokens):
    return pos_tag(sentence_tokens)


def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word)


def lemmatize(word, tag=None):
    lemmatizer = WordNetLemmatizer()
    if tag is None:
        return lemmatizer.lemmatize(word)
    else:
        return lemmatizer.lemmatize(word, tag)


if __name__ == '__main__':
    dataset = load_data()

    sentences = dataset['tweet'].values

    tokens = word_tokenize(sentences[0])
    print(f'Sentence: {sentences[0]}')
    print(f'Tokens: {tokens}')

    pos_tags = find_pos_tag(tokens)
    print(f'POS tags: {pos_tags}')

    word = 'worse'
    tag = 'a'
    print(f'Word: {word}')
    print(f'Stem: {stem(word)}')
    print(f'Lemma without POS tag: {lemmatize(word)}')
    print(f'Lemma with POS tag: {lemmatize(word, tag)}')
