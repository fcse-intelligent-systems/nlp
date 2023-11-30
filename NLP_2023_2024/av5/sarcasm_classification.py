import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util


def load_data():
    return pd.read_csv('../data/train.En.csv',
                       usecols=['tweet', 'sarcastic']).dropna()[:100]


if __name__ == '__main__':
    dataset = load_data()

    samples = dataset['tweet'].values.tolist()
    labels = dataset['sarcastic'].values.tolist()

    train_samples, test_samples, \
        train_labels, test_labels = train_test_split(samples, labels,
                                                     test_size=0.2, random_state=0,
                                                     stratify=labels)

    model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

    prompt_template = 'Classify the text into sarcastic or non-sarcastic: '

    # zero-shot
    for sample, label in zip(test_samples, test_labels):
        prompt = f'{prompt_template}{sample}'

        input_data = tokenizer(prompt, return_tensors='pt')
        input_ids = input_data.input_ids

        output = model.generate(input_ids)
        pred_label = tokenizer.decode(output[0])

        print(f'Prompt: {prompt}\n'
              f'Sample: {sample}\n'
              f'True label: {label}\n'
              f'Predicted label: {pred_label}\n')

    # few-shot
    embedding_model = SentenceTransformer('all-distilroberta-v1')
    embeddings = embedding_model.encode(train_samples,
                                        batch_size=64,
                                        show_progress_bar=True)

    for sample, label in zip(test_samples, test_labels):
        sentence_embeddings = embedding_model.encode(sample,
                                                     convert_to_tensor=True)
        results = util.semantic_search(sentence_embeddings, embeddings, top_k=1)
        example_index = results[0][0]['corpus_id']

        example_text = train_samples[example_index]
        example_label = 'sarcastic' if train_labels[example_index] == 1 else 'non-sarcastic'
        example = f'Text: {example_text}\nCategory: {example_label}'

        prompt = f'{example}\nBased on the above example, classify the text into sarcastic or non-sarcastic: {sample}'

        input_data = tokenizer(prompt, return_tensors='pt')
        input_ids = input_data.input_ids

        output = model.generate(input_ids)
        pred_label = tokenizer.decode(output[0])

        print(f'Prompt: {prompt}\n'
              f'Sample: {sample}\n'
              f'True label: {label}\n'
              f'Predicted label: {pred_label}\n')
