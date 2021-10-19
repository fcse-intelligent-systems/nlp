import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data():
    return pd.read_csv('../data/train.En.csv', usecols=['tweet', 'sarcastic']).dropna()


if __name__ == '__main__':
    dataset = load_data()

    sentences = dataset['tweet'].values
    labels = dataset['sarcastic'].values

    x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.1,
                                                        random_state=0, stratify=labels)

    tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), stop_words={'english'})
    x_train_tfidf = tfidf.fit_transform(x_train)
    x_test_tfidf = tfidf.transform(x_test)

    print(f'Features: {tfidf.get_feature_names()}')

    classifier = RandomForestClassifier(n_estimators=2500, random_state=0)
    classifier.fit(x_train_tfidf, y_train)

    predictions = classifier.predict(x_test_tfidf)
    print(f'Predictions: {predictions}')

    acc = accuracy_score(predictions, y_test)
    print(f'Accuracy: {acc}')
