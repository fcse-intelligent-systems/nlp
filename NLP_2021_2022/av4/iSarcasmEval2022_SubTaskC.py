import pandas as pd
import numpy as np
from nltk import word_tokenize
from scripts.word_embeddings import load_embedding_weights
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split


def load_data():
    return pd.read_csv('../data/train.En.csv', usecols=['tweet', 'rephrase']).dropna()


def tokenize(data):
    data['tweet_tokens'] = data['tweet'].apply(lambda x: word_tokenize(x.lower()))
    data['rephrase_tokens'] = data['rephrase'].apply(lambda x: word_tokenize(x.lower()))


def create_vocabulary(sentence_tokens):
    vocabulary = set()
    for tokens in sentence_tokens:
        vocabulary.update(tokens)

    vocabulary = list(vocabulary)
    word_to_id = {word: index for word, index in zip(vocabulary, range(len(vocabulary)))}
    return vocabulary, word_to_id


def create_train_test_data(sentences, rephrases):
    sent1, sent2, labels = [], [], []
    for sentence, rephrase in zip(sentences, rephrases):
        p = np.random.randint(2)
        if p == 0:
            sent1.append(sentence)
            sent2.append(rephrase)
            labels.append(0)
        else:
            sent1.append(rephrase)
            sent2.append(sentence)
            labels.append(1)
    return sent1, sent2, labels


if __name__ == '__main__':
    dataset = load_data()

    tokenize(dataset)

    sentences = dataset['tweet_tokens'].values
    rephrases = dataset['rephrase_tokens'].values

    vocabulary, word_to_id = create_vocabulary(np.concatenate((sentences, rephrases)))

    embeddings = load_embedding_weights(vocabulary, 50, 'glove')

    dataset['tweet_indices'] = dataset['tweet_tokens'].apply(lambda x: np.array([word_to_id[i] for i in x]))
    dataset['rephrase_indices'] = dataset['rephrase_tokens'].apply(lambda x: np.array([word_to_id[i] for i in x]))

    sentence_indices = dataset['tweet_indices'].values
    rephrase_indices = dataset['rephrase_indices'].values

    padded_sentences = pad_sequences(sentence_indices, 10)
    padded_rephrases = pad_sequences(rephrase_indices, 10)

    sentences1, sentences2, labels = create_train_test_data(padded_sentences, padded_rephrases)

    x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(sentences1, sentences2, labels,
                                                                             test_size=0.1, random_state=0,
                                                                             stratify=labels)

    input1 = Input(shape=(10,))
    x1 = Embedding(input_dim=len(vocabulary), output_dim=50, weights=[embeddings], trainable=False)(input1)
    x1 = LSTM(128)(x1)

    input2 = Input(shape=(10,))
    x2 = Embedding(input_dim=len(vocabulary), output_dim=50, weights=[embeddings], trainable=False)(input2)
    x2 = LSTM(128)(x2)

    x = Concatenate()([x1, x2])

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input1, input2], outputs=output)

    model.compile(optimizer=Adam(learning_rate=0.01), loss=binary_crossentropy, metrics=['accuracy'])

    model.fit([np.array(x1_train), np.array(x2_train)],
              np.array(y_train),
              batch_size=32, epochs=15, verbose=2)

    model.evaluate([np.array(x1_test), np.array(x2_test)],
                   np.array(y_test))

