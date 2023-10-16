from gensim import downloader

if __name__ == '__main__':
    embeddings = downloader.load('glove-twitter-50')

    embeddings.most_similar('student')

    # Paris - France + Italy
    embeddings.most_similar(positive=['paris', 'italy'], negative=['france'])

    # King - Man + Woman
    embeddings.most_similar(positive=['king', 'woman'], negative=['man'])
