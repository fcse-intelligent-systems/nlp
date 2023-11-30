from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

if __name__ == '__main__':
    model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

    prompt = 'Translate the text from English to Spanish: I loved that restaurant!'
    input_data = tokenizer(prompt, return_tensors='pt')
    input_ids = input_data.input_ids

    output = model.generate(input_ids)
    translation = tokenizer.decode(output[0])

    print('English sentence: I loved that restaurant!')
    print(f'Spanish sentence: {translation}')
