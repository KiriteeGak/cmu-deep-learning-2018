import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.layers.core import Activation
from keras.losses import categorical_crossentropy

from hw2.masked_batch_1 import bucketize_points


def lstm_architecture(size):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(size, 40)))
    model.add(TimeDistributed(Dense(139)))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    return model


def make_data():
    train_data_raw = np.load("/home/kiriteegak/Desktop/"
                             "github-general/cmu-deep-learning-2018/"
                             "resources/HW1P2/train.npy",
                             encoding="bytes")
    train_data_labels = np.load("/home/kiriteegak/Desktop/"
                                "github-general/cmu-deep-learning-2018/"
                                "resources/HW1P2/train_labels.npy",
                                encoding="bytes")
    max_size = max([_.shape[0] for _ in train_data_raw])
    padded_sequences, padded_labels = [], []
    for i, _ in enumerate(train_data_raw):
        seq_, label = bucketize_points(_, to_one=max_size, masks=False, category=train_data_labels[i])
        padded_sequences.append(seq_), padded_labels.append(label)
    return np.array(padded_sequences), np.array(padded_labels), max_size


if __name__ == '__main__':
    seq, labels, size_ = make_data()
    np.save("/home/kiriteegak/Desktop/github-general/cmu-deep-learning-2018/hw3/data/hw2_modified/sequences.npy", seq)
    np.save("/home/kiriteegak/Desktop/github-general/cmu-deep-learning-2018/hw3/data/hw2_modified/labels.npy", labels)
    model_ = lstm_architecture(size_)
    model_.fit(x=seq, y=labels, batch_size=32, shuffle=True, epochs=100)
