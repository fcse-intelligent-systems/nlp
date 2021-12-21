from datasets import load_dataset
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast, TFBertModel
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error


def calculate_end_index(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text'][0]
        start_idx = answer['answer_start'][0]
        end_idx = start_idx + len(gold_text)

        answer['text'] = gold_text

        if context[start_idx:end_idx] == gold_text:
            answer['answer_start'] = start_idx
            answer['answer_end'] = end_idx
        elif context[start_idx - 1:end_idx - 1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1
        elif context[start_idx - 2:end_idx - 2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2


if __name__ == '__main__':
    dataset = load_dataset('squad')

    train = pd.DataFrame().from_dict(dataset['train'])
    test = pd.DataFrame().from_dict(dataset['validation'])

    train_contexts = train['context'].values.tolist()
    train_questions = train['question'].values.tolist()
    train_answers = train['answers'].values.tolist()

    test_contexts = test['context'].values.tolist()
    test_questions = test['question'].values.tolist()
    test_answers = test['answers'].values.tolist()

    calculate_end_index(train_answers, train_contexts)
    calculate_end_index(test_answers, test_contexts)

    bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    train_encodings = bert_tokenizer(train_contexts, train_questions, max_length=100,
                                     truncation=True, pad_to_max_length=True)
    train_input_ids = train_encodings['input_ids']
    train_attention_masks = train_encodings['attention_mask']
    train_start_positions, train_end_positions = [], []
    for i in range(len(train_answers)):
        train_start_positions.append(train_encodings.char_to_token(i, train_answers[i]['answer_start']))
        train_end_positions.append(train_encodings.char_to_token(i, train_answers[i]['answer_end']))
        if train_start_positions[-1] is None:
            train_start_positions[-1] = bert_tokenizer.model_max_length
        if train_end_positions[-1] is None:
            train_end_positions[-1] = bert_tokenizer.model_max_length

    test_encodings = bert_tokenizer(test_contexts, test_questions, max_length=100,
                                    truncation=True, pad_to_max_length=True)
    test_input_ids = test_encodings['input_ids']
    test_attention_masks = test_encodings['attention_mask']
    test_start_positions, test_end_positions = [], []
    for i in range(len(test_answers)):
        test_start_positions.append(test_encodings.char_to_token(i, test_answers[i]['answer_start']))
        test_end_positions.append(test_encodings.char_to_token(i, test_answers[i]['answer_end']))
        if test_start_positions[-1] is None:
            test_start_positions[-1] = bert_tokenizer.model_max_length
        if test_end_positions[-1] is None:
            test_end_positions[-1] = bert_tokenizer.model_max_length

    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    input_ids = Input(shape=(100,), name='input_token', dtype='int32')
    att_masks = Input(shape=(100,), name='masked_token', dtype='int32')

    bert_in = bert_model(input_ids, attention_mask=att_masks)[1]

    start = Dense(1, activation='relu', name='start')(bert_in)
    end = Dense(1, activation='relu', name='end')(bert_in)

    bert_qa_model = Model(inputs=[input_ids, att_masks],
                          outputs=[start, end])

    bert_qa_model.compile(optimizer=Adam(learning_rate=0.01),
                          loss=mean_squared_error,
                          metrics=['accuracy'])

    bert_qa_model.fit([np.array(train_input_ids), np.array(train_attention_masks)],
                      [np.array(train_start_positions), np.array(train_end_positions)],
                      epochs=50)

    bert_qa_model.evaluate([np.array(test_input_ids), np.array(test_attention_masks)],
                           [np.array(test_start_positions), np.array(test_end_positions)])
