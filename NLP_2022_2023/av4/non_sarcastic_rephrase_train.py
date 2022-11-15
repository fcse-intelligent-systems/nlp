import pandas as pd
import numpy as np
from nltk import word_tokenize
from scripts.word_embeddings import load_embedding_weights
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def load_data():
    return pd.read_csv('../data/train.En.csv', usecols=['tweet', 'rephrase']).dropna()


def tokenize(data):
    data['tweet_tokens'] = data['tweet'].apply(lambda x: word_tokenize(x.lower()))
    data['rephrase_tokens'] = data['rephrase'].apply(lambda x: word_tokenize(x.lower()))


def append_start_end(data):
    data['tweet_tokens'] = data['tweet_tokens'].apply(lambda x: np.concatenate((['<START>'], x, ['<END>'])))
    data['rephrase_tokens'] = data['rephrase_tokens'].apply(lambda x: np.concatenate((['<START>'], x, ['<END>'])))


def create_vocabulary(sentence_tokens):
    vocabulary = set()
    for tokens in sentence_tokens:
        vocabulary.update(tokens)
    vocabulary = list(vocabulary)
    word_to_id = {word: index for word, index in zip(vocabulary, range(len(vocabulary)))}
    id_to_word = {index: word for word, index in zip(vocabulary, range(len(vocabulary)))}
    return vocabulary, word_to_id, id_to_word


def create_train_data(sentences, rephrases):
    input_sentences, input_rephrases, next_words = [], [], []
    for sentence, rephrase in zip(sentences, rephrases):
        # 3 5 7 9

        # 3       5
        # 3 5     7
        # 3 5 7   9
        for i in range(1, len(rephrase)):
            input_sentences.append(sentence)
            input_rephrases.append(rephrase[:i])
            next_words.append(rephrase[i])
    return input_sentences, input_rephrases, next_words


def create_model(padding_size, vocabulary_size, embedding_size, embeddings):
    encoder_inputs = Input(shape=(padding_size,))
    encoder_embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_size,
                                  weights=[embeddings], trainable=False)(encoder_inputs)
    encoder = LSTM(128, return_state=True)
    _, state_h, state_c = encoder(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(padding_size,))
    decoder_embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_size,
                                  weights=[embeddings], trainable=False)(decoder_inputs)
    decoder = LSTM(128, return_state=True)
    decoder_outputs, _, _ = decoder(decoder_embedding, initial_state=encoder_states)

    decoder_outputs = Dense(vocabulary_size, activation='softmax')(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer=Adam(learning_rate=0.01), loss=categorical_crossentropy)

    return model


if __name__ == '__main__':
    dataset = load_data()

    tokenize(dataset)

    append_start_end(dataset)

    sentences = dataset['tweet_tokens'].values
    rephrases = dataset['rephrase_tokens'].values

    vocabulary, word_to_id, id_to_word = create_vocabulary(np.concatenate((sentences, rephrases)))

    embeddings = load_embedding_weights(vocabulary, 50, 'glove')

    dataset['tweet_indices'] = dataset['tweet_tokens'].apply(lambda x: np.array([word_to_id[i] for i in x]))
    sentence_indices = dataset['tweet_indices'].values

    dataset['rephrase_indices'] = dataset['rephrase_tokens'].apply(lambda x: np.array([word_to_id[i] for i in x]))
    rephrase_indices = dataset['rephrase_indices'].values

    train_sentences, test_sentences, train_rephrases, test_rephrases = train_test_split(sentence_indices,
                                                                                        rephrase_indices,
                                                                                        test_size=0.1,
                                                                                        random_state=0)

    input_sentences, input_rephrases, next_words = create_train_data(train_sentences, train_rephrases)

    padded_sentences = pad_sequences(input_sentences, 10)
    padded_rephrases = pad_sequences(input_rephrases, 10)

    label_binarizer = LabelBinarizer()
    label_binarizer.fit_transform(list(word_to_id.values()))
    next_words = label_binarizer.transform(next_words)

    model = create_model(10, len(vocabulary), 50, embeddings)

    model.fit([np.array(padded_sentences), np.array(padded_rephrases)],
              np.array(next_words),
              batch_size=64, epochs=500, verbose=2)

    model.save_weights('../models/EncoderDecoder.h5')
