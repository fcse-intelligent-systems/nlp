import os
import _pickle as pickle
import numpy as np


def load_embeddings(file_name, vocabulary):
    """
    Loads word embeddings from the file with the given name.
    :param file_name: name of the file containing word embeddings
    :type file_name: str
    :param vocabulary: captions vocabulary
    :type vocabulary: numpy.array
    :return: word embeddings
    :rtype: dict
    """
    embeddings = dict()
    with open(file_name, 'r', encoding='utf-8') as doc:
        line = doc.readline()
        while line != '':
            line = line.rstrip('\n').lower()
            parts = line.split(' ')
            vals = np.array(parts[1:], dtype=np.float)
            if parts[0] in vocabulary:
                embeddings[parts[0]] = vals
            line = doc.readline()
    return embeddings


def load_embedding_weights(vocabulary, embedding_size, embedding_type):
    """
    Creates and loads embedding weights.
    :param vocabulary: vocabulary
    :type vocabulary: numpy.array
    :param embedding_size: embedding size
    :type embedding_size: int
    :param embedding_type: type of the pre-trained embeddings
    :type embedding_type: string
    :return: embedding weights
    :rtype: numpy.array
    """
    if os.path.exists(f'../data/embedding_matrix_{embedding_type}_{embedding_size}.pkl'):
        with open(f'../data/embedding_matrix_{embedding_type}_{embedding_size}.pkl', 'rb') as f:
            embedding_matrix = pickle.load(f)
    else:
        print('Creating embedding weights...')
        if embedding_type == 'glove':
            embeddings = load_embeddings(f'../data/glove.6B.{embedding_size}d.txt', vocabulary)
        else:
            embeddings = load_embeddings(f'../data/word2vecSG.iSarcasamEval.{embedding_size}d.txt', vocabulary)
        embedding_matrix = np.zeros((len(vocabulary), embedding_size))
        for i in range(len(vocabulary)):
            if vocabulary[i] in embeddings.keys():
                embedding_matrix[i] = embeddings[vocabulary[i]]
            else:
                embedding_matrix[i] = np.random.standard_normal(embedding_size)
        with open(f'../data/embedding_matrix_{embedding_type}_{embedding_size}.pkl', 'wb') as f:
            pickle.dump(embedding_matrix, f)
    return embedding_matrix
