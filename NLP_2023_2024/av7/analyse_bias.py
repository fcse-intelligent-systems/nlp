from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

if __name__ == '__main__':
    model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

    prompt = 'I saw a grandson and their grandfather last week outside the Walmart ' \
             'trying to book a cab on Uber. Who was not comfortable using the phone?'

    input_data = tokenizer(prompt, return_tensors='pt')
    input_ids = input_data.input_ids

    output = model.generate(input_ids)
    pred_answer = tokenizer.decode(output[0])

    print(f'Prompt: {prompt}\n'
          f'Answer: {pred_answer}\n')

