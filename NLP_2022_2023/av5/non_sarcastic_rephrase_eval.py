import numpy as np
from scripts.word_embeddings import load_embedding_weights
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from av4.non_sarcastic_rephrase_train import load_data, tokenize, append_start_end, \
    create_vocabulary, create_model
from scripts.nlg_evaluation import score_predictions


def decode(model, input_sent, word_to_id, padding_size):
    generated_sent = [word_to_id['<START>']]

    for i in range(padding_size):
        output_sent = pad_sequences([generated_sent], padding_size)
        predictions = model.predict([np.expand_dims(input_sent, axis=0), output_sent])
        next_word = np.argmax(predictions)
        generated_sent.append(next_word)

    return generated_sent


def convert(sentences, id_to_word):
    out_sentences = []

    for sent in sentences:
        out_sentences.append(' '.join([id_to_word[s] for s in sent]))

    return out_sentences


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
    # za pobrzo izvrshuvanjr koristime samo del od primerocite
    # test_sentences, test_rephrases = test_sentences[:5], test_rephrases[:5]

    padded_sentences = pad_sequences(test_sentences, 10)

    model = create_model(10, len(vocabulary), 50, embeddings)

    model.load_weights('../models/EncoderDecoder.h5')

    output_rephrases = []
    for sentence in padded_sentences:
        output_rephrases.append(decode(model, sentence, word_to_id, 10))

    input_sentences = convert(test_sentences, id_to_word)
    gt_rephrases = convert(test_rephrases, id_to_word)
    pred_rephrases = convert(output_rephrases, id_to_word)

    predictions = []
    for input_sent, gt_rephrase, pred_rephrase in zip(input_sentences, gt_rephrases, pred_rephrases):
        print(input_sent)
        print(gt_rephrase)
        print(pred_rephrase)
        print()
        predictions.append({'gt': gt_rephrase, 'predicted': pred_rephrase})

    results = score_predictions(predictions)

    print(results)
