from datasets import load_dataset
from transformers import BertTokenizerFast, TFBertModel
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error

if __name__ == '__main__':
    dataset = load_dataset('commonsense_qa')

    questions = dataset['train']['question']
    choices = [choice['text'] for choice in dataset['train']['choices']]
    answers = dataset['train']['answerKey']

    sequences = []
    for question, choice in zip(questions, choices):
        sequences.extend([f'{question} - {c}' for c in choice])

    print(len(sequences))

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    tokens = tokenizer(sequences, truncation=True, max_length=100)
    tokens = {k: [[e for el in v[i: i + 5] for e in el] for i in range(0, len(v), 5)] for k, v in tokens.items()}

    print(len(tokens['input_ids']))

    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    input_ids = Input(shape=(100,), name='input_token', dtype='int32')
    att_masks = Input(shape=(100,), name='masked_token', dtype='int32')

    bert_in = bert_model(input_ids, attention_mask=att_masks)[1]

    answer = Dense(5, activation='relu', name='answer')(bert_in)

    model = Model(inputs=[input_ids, att_masks],
                  outputs=[answer])

    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss=mean_squared_error,
                  metrics=['accuracy'])
