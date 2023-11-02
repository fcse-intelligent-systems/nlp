import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import TFBertModel, TFBertForSequenceClassification, BertTokenizer
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy


def load_data():
    return pd.read_csv('../data/train.En.csv',
                       usecols=['tweet', 'sarcastic']).dropna()[:100]


if __name__ == '__main__':
    dataset = load_data()

    dataset_x = dataset['tweet'].values.tolist()
    dataset_y = dataset['sarcastic'].values.tolist()

    train_x, test_x, train_y, test_y = train_test_split(dataset_x, dataset_y,
                                                        test_size=0.1, random_state=0,
                                                        stratify=dataset_y)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_input_ids, train_attention_mask = [], []

    for sample in train_x:
        result = bert_tokenizer.encode_plus(sample, max_length=10,
                                            pad_to_max_length=True, truncation=True)
        train_input_ids.append(result['input_ids'])
        train_attention_mask.append(result['attention_mask'])

    test_input_ids, test_attention_mask = [], []

    for sample in test_x:
        result = bert_tokenizer.encode_plus(sample, max_length=10,
                                            pad_to_max_length=True, truncation=True)
        test_input_ids.append(result['input_ids'])
        test_attention_mask.append(result['attention_mask'])

    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                            num_labels=2)

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=binary_crossentropy)

    model.fit([np.array(train_input_ids), np.array(train_attention_mask)],
              np.array(train_y),
              epochs=5)

    model.evaluate([np.array(test_input_ids), np.array(test_attention_mask)],
                   test_y)

    y_pred = model.predict([test_input_ids, test_attention_mask])
