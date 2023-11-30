from transformers import pipeline

if __name__ == '__main__':
    model = pipeline('text-generation', model='gpt2')

    prompt = 'Hello, I\'m a language model'

    result = model(prompt, max_length=15)
    print(result)

    result = model(prompt, max_length=15, return_full_text=False)
    print(result)

    result = model(prompt, max_length=15, return_full_text=False)
    print(result[0]['generated_text'])
