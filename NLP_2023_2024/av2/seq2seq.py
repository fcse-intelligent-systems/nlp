from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy


def create_train_data(sentences, translations):
    input_sentences, input_translations, next_words = [], [], []
    for sentence, rephrase in zip(sentences, translations):
        for i in range(1, len(rephrase)):
            input_sentences.append(sentence)
            input_translations.append(rephrase[:i])
            next_words.append(rephrase[i])
    return input_sentences, input_translations, next_words


def create_model(padding_size, vocabulary_size_en, vocabulary_size_es, embedding_size):
    encoder_inputs = Input(shape=(padding_size,))
    encoder_embedding = Embedding(input_dim=vocabulary_size_en,
                                  output_dim=embedding_size)(encoder_inputs)
    encoder = LSTM(128, return_state=True)
    _, state_h, state_c = encoder(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(padding_size,))
    decoder_embedding = Embedding(input_dim=vocabulary_size_es, output_dim=embedding_size,
                                  trainable=False)(decoder_inputs)
    decoder = LSTM(128, return_state=True)
    decoder_outputs, _, _ = decoder(decoder_embedding,
                                    initial_state=encoder_states)

    decoder_outputs = Dense(vocabulary_size_es, activation='softmax')(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs],
                  decoder_outputs)

    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss=categorical_crossentropy)

    return model
