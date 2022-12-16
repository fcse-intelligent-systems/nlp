import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy


def load_data():
	return pd.read_csv('../data/train.En.csv', usecols=['tweet', 'rephrase']).dropna()


def create_train_test_data(s_in_ids, s_att_masks, r_in_ids, r_att_masks):
	# sent1_in_ids, sent1_att_masks, sent2_in_ids, sent2_att_masks, labels = [], [], [], [], []
	sent_in_ids, sent_att_masks, labels = [], [], []
	for s_ids, s_masks, r_ids, r_masks in zip(s_in_ids, s_att_masks, r_in_ids, r_att_masks):
		p = np.random.randint(2)
		if p == 0:
			sent_in_ids.append(s_ids+r_ids)
			sent_att_masks.append(s_masks+r_masks)
			labels.append(0)
		else:
			sent_in_ids.append(r_ids+s_ids)
			sent_att_masks.append(r_masks+s_masks)
			labels.append(1)
	return sent_in_ids, sent_att_masks, labels


if __name__ == '__main__':
	dataset = load_data()

	sentences = dataset['tweet'].values
	rephrases = dataset['rephrase'].values

	bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	sentence_input_ids, sentence_attention_masks = [], []
	rephrase_input_ids, rephrase_attention_masks = [], []
	for sentence, rephrase in zip(sentences, rephrases):
		sentence_tokens = bert_tokenizer.encode_plus(sentence, max_length=10,
													 pad_to_max_length=True, truncation=True)
		sentence_input_ids.append(sentence_tokens['input_ids'])
		sentence_attention_masks.append(sentence_tokens['attention_mask'])

		rephrase_tokens = bert_tokenizer.encode_plus(rephrase, max_length=10,
													 pad_to_max_length=True, truncation=True)
		rephrase_input_ids.append(rephrase_tokens['input_ids'])
		rephrase_attention_masks.append(rephrase_tokens['attention_mask'])

	input_ids, attention_masks, labels = create_train_test_data(sentence_input_ids,
																sentence_attention_masks,
																rephrase_input_ids,
																rephrase_attention_masks)

	ids_train, ids_test, masks_train, masks_test, y_train, y_test = train_test_split(input_ids, attention_masks,
																					 labels, test_size=0.1,
																					 random_state=0, stratify=labels)

	model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

	model.compile(optimizer=Adam(learning_rate=0.01), loss=binary_crossentropy, metrics=['accuracy'])

	model.fit([np.array(ids_train), np.array(masks_train)],
			  np.array(y_train),
			  epochs=15)

	model.evaluate([np.array(ids_test), np.array(masks_test)],
				   np.array(y_test))
