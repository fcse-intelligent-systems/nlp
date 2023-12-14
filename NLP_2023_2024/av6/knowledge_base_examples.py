from datasets import load_dataset

if __name__ == '__main__':
    kb = load_dataset('generics_kb')  # ascent_kb

    print(kb['train'][0])

    sentences = kb['train']['generic_sentence']

    print(len(sentences))

    print(sentences[:15])

    question = 'What do you call a group of dolphins?'
    knowledge = 'A group of dolphins is called a \'school\' or a \'pod\'.'
    knowledge_augmented_sequence = f'{question} | {knowledge}'

    print(knowledge_augmented_sequence)
