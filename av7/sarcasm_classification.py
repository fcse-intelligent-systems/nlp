import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy


def load_data():
    return pd.read_csv('../data/train.En.csv', usecols=['tweet', 'sarcastic']).dropna()


if __name__ == '__main__':
    dataset = load_data()

    sentences = dataset['tweet'].values
    labels = dataset['sarcastic'].values

    x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.1, random_state=0,
                                                        stratify=labels)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_input_ids, train_attention_masks, train_outputs = [], [], []
    for sentence, label in zip(x_train, y_train):
        sentence_tokens = bert_tokenizer.encode_plus(sentence, max_length=10,
                                                     pad_to_max_length=True, truncation=True)
        train_input_ids.append(sentence_tokens['input_ids'])
        train_attention_masks.append(sentence_tokens['attention_mask'])
        train_outputs.append(label)

    test_input_ids, test_attention_masks, test_outputs = [], [], []
    for sentence, label in zip(x_test, y_test):
        sentence_tokens = bert_tokenizer.encode_plus(sentence, max_length=10,
                                                     pad_to_max_length=True, truncation=True)
        test_input_ids.append(sentence_tokens['input_ids'])
        test_attention_masks.append(sentence_tokens['attention_mask'])
        test_outputs.append(label)

    bert_classification_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    bert_classification_model.compile(optimizer=Adam(learning_rate=0.01),
                                      loss=binary_crossentropy,
                                      metrics=['accuracy'])

    bert_classification_model.fit([np.array(train_input_ids), np.array(train_attention_masks)],
                                  np.array(train_outputs),
                                  epochs=15)

    bert_classification_model.evaluate([np.array(test_input_ids), np.array(test_attention_masks)],
                                       np.array(test_outputs))
