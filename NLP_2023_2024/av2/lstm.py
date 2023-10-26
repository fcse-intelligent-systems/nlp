from tensorflow.keras.layers import LSTM, Embedding, Input
from tensorflow.keras.models import Sequential, Model

if __name__ == '__main__':
    model = Sequential()

    # 1
    model.add(LSTM())

    # 2
    model.add(Embedding())
    model.add(LSTM())

    # 3
    input_layer = Input()
    embedding_layer = Embedding()(input_layer)
    lstm_layer = LSTM()(embedding_layer)
