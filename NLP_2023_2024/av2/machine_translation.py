import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences
from seq2seq import create_train_data, create_model


def load_data():
    data = pd.read_csv('../data/en_es_corpus.txt',
                       sep='\t', header=None)[:100]
    data = data[[0, 1]]
    data.columns = ['EN', 'ES']
    return data


def tokenize(data):
    data['EN_tokens'] = data['EN'].apply(lambda x: word_tokenize(x.lower(),
                                                                 language='english'))
    data['ES_tokens'] = data['ES'].apply(lambda x: word_tokenize(x.lower(),
                                                                 language='spanish'))


def append_start_end_token(data):
    data['EN_tokens'] = data['EN_tokens'].apply(lambda x: np.concatenate((['<START>'], x, ['<END>'])))
    data['ES_tokens'] = data['ES_tokens'].apply(lambda x: np.concatenate((['<START>'], x, ['<END>'])))


def create_vocabulary(sentences):
    vocab = set()
    for sentence in sentences:
        vocab.update(sentence)
    vocab = list(vocab)
    w_to_i = {word: index for index, word in enumerate(vocab)}
    i_to_w = {index: word for index, word in enumerate(vocab)}

    return vocab, w_to_i, i_to_w


def map_to_index(data, w_to_i_en, w_to_i_es):
    data['EN_index'] = data['EN_tokens'].apply(lambda x: [w_to_i_en[word] for word in x])
    data['ES_index'] = data['ES_tokens'].apply(lambda x: [w_to_i_es[word] for word in x])


if __name__ == '__main__':
    dataset = load_data()

    tokenize(dataset)

    append_start_end_token(dataset)

    vocab_en, word_to_id_en, id_to_word_en = create_vocabulary(dataset['EN_tokens'].values.tolist())
    vocab_es, word_to_id_es, id_to_word_es = create_vocabulary(dataset['ES_tokens'].values.tolist())

    map_to_index(dataset, word_to_id_en, word_to_id_es)

    indices_en = dataset['EN_index'].values.tolist()
    indices_es = dataset['ES_index'].values.tolist()

    train_en, test_en, train_es, test_es = train_test_split(indices_en, indices_es,
                                                            test_size=0.1, random_state=0)

    # I care.	Me preocupo.

    # [1, 2]   # [4, 5, 6]
    # [1, 2]  <START>        4
    # [1, 2]  <START> 4      5
    # [1, 2]  <START> 4 5    6

    input_en, input_es, output_es = create_train_data(train_en, train_es)

    model = create_model(10, len(vocab_en), len(vocab_es), 50)

    input_en_padded = pad_sequences(input_en, 10)
    input_es_padded = pad_sequences(input_es, 10)

    label_binarizer = LabelBinarizer()
    label_binarizer.fit_transform(list(word_to_id_es.values()))
    next_words = label_binarizer.transform(output_es)

    model.fit([input_en_padded, input_es_padded],
              next_words,
              epochs=5, batch_size=16)

    model.save_weights('../models/EncoderDecoder.h5')
