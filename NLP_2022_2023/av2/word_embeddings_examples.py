import gensim.downloader

if __name__ == '__main__':
	glove_embeddings = gensim.downloader.load('glove-twitter-50')

	glove_embeddings.most_similar('student')

	# King - Woman + Man
	glove_embeddings.most_similar(positive=['woman', 'king'], negative=['man'])

	# Paris – France + Italy
	glove_embeddings.most_similar(positive=['paris', 'italy'], negative=['france'])

	# Madrid – Spain + France
	glove_embeddings.most_similar(positive=['madrid', 'spain'], negative=['france'])

	# Windows – Microsoft + Google
	glove_embeddings.most_similar(positive=['windows', 'microsoft'], negative=['google'])

	# Sushi - Germany + Japan
	glove_embeddings.most_similar(positive=['sushi', 'germany'], negative=['japan'])
