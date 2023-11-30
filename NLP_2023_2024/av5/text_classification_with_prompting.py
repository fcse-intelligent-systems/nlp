from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

if __name__ == '__main__':
    model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

    prompt = 'Classify the text into positive or negative: I loved that restaurant!'
    input_data = tokenizer(prompt, return_tensors='pt')
    input_ids = input_data.input_ids

    output = model.generate(input_ids)
    label = tokenizer.decode(output[0])

    print('Sentence: I loved that restaurant!')
    print(f'Label: {label}')

    """
    Text: Example 1
    Category: C1
    Text: Example 1
    Category: C2
    <Instruction> ...
    """

    sample = 'I loved that restaurant!'
    example_1 = 'Text: I like the food in that restaurant!\nCategory: positive'
    example_2 = 'Text: I hated that restaurant!\nCategory: negative'

    prompt = f'{example_1}\n{example_2}\nBased on the above examples, classify the text into positive or negative: {sample}'
    print(prompt)

    input_data = tokenizer(prompt, return_tensors='pt')
    input_ids = input_data.input_ids

    output = model.generate(input_ids)
    label = tokenizer.decode(output[0])

    print(f'Sentence: {sample}')
    print(f'Label: {label}')
