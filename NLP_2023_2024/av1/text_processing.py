import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer

if __name__ == '__main__':
    sentence = 'Tokenization is easy, they said! Just split on whitespace, they said!'

    print(sent_tokenize(sentence))
    print(word_tokenize(sentence))

    print(pos_tag(word_tokenize(sentence)))

    print(nltk.help.upenn_tagset())

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    print(stemmer.stem('stripes'))
    print(lemmatizer.lemmatize('stripes', 'v'))
    print(lemmatizer.lemmatize('stripes', 'n'))

    print(stemmer.stem('worse'))
    print(lemmatizer.lemmatize('worse'))
    print(lemmatizer.lemmatize('worse', 'a'))
