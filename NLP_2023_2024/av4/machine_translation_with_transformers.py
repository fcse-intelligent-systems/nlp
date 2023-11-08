import pandas as pd
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from seq2seq import create_transformers_train_data, decode_with_transformer
from evaluate import load


def load_data():
    data = pd.read_csv('../data/en_es_corpus.txt',
                       sep='\t', header=None)[:100]
    data = data[[0, 1]]
    data.columns = ['EN', 'ES']
    return data


if __name__ == '__main__':
    dataset = load_data()

    prefix = 'translate from English to Spanish: '
    model_name = 't5-small'

    sentences_en = [prefix + sentence for sentence in dataset['EN'].values.tolist()]
    sentences_es = dataset['ES'].values.tolist()

    train_en, test_en, train_es, test_es = train_test_split(sentences_en, sentences_es,
                                                            test_size=0.1, random_state=0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_set = create_transformers_train_data(train_en, train_es, tokenizer)

    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           model=model_name,
                                           return_tensors='tf')

    train_set = model.prepare_tf_dataset(train_set, collate_fn=data_collator)

    model.compile(Adam(learning_rate=0.01))

    model.fit(train_set, epochs=5)

    output_es = []
    for sentence in test_en:
        pred = decode_with_transformer(sentence, tokenizer, model)
        output_es.append(pred)

    input_en = test_en
    output_es_gt = test_es
    output_es_pred = output_es

    for in_en, gt_es, pred_es in zip(input_en, output_es_gt, output_es_pred):
        print(f'Input sentence: {in_en}')
        print(f'GT translation: {gt_es}')
        print(f'Pred translation: {pred_es}')

    metric = load('bleu')
    results = metric.compute(predictions=output_es_pred, references=output_es_gt)
    score = results['bleu']

    print(f'BLEU score: {score}')
