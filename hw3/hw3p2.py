import numpy as np

from keras.backend.tensorflow_backend import eval
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.layers.core import Activation
from keras.losses import categorical_crossentropy
from keras.models import load_model

from hw2.masked_batch_1 import bucketize_points


def lstm_architecture(size):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(size, 40)))
    model.add(TimeDistributed(Dense(139)))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    return model


def make_data(data_size=None, train_data_raw=None, train_data_labels=None):
    for st in range(0, data_size, 32):
        padded_sequences, padded_labels = [], []
        for i, _ in enumerate(train_data_raw[st:st+32]):
            seq_, label = bucketize_points(_, to_one=max_size, masks=False, category=train_data_labels[i])
            padded_sequences.append(seq_), padded_labels.append(label)
        yield np.array(padded_sequences), np.array(padded_labels, dtype='float32')


if __name__ == '__main__':
    train_data_raw = np.load("/home/kiriteegak/Desktop/"
                             "github-general/cmu-deep-learning-2018/"
                             "resources/HW1P2/train.npy",
                             encoding="bytes")
    # train_data_labels = np.load("/home/kiriteegak/Desktop/"
    #                             "github-general/cmu-deep-learning-2018/"
    #                             "resources/HW1P2/train_labels.npy",
    #                             encoding="bytes")
    max_size = max([_.shape[0] for _ in train_data_raw])
    # data_size = train_data_labels.shape[0]
    #
    # model_ = lstm_architecture(max_size)
    # model_ = load_model("/home/kiriteegak/Desktop/github-general/cmu-deep-learning-2018/"
    #                     "hw3/models/1_layer_lstm_p2_modified_epoch_1")
    # for _ in range(8):
    #     model_.fit_generator(make_data(),
    #                          shuffle=True,
    #                          steps_per_epoch=int(train_data_labels.shape[0]/32))
    #     model_.save("/home/kiriteegak/Desktop/github-general/cmu-deep-learning-2018/"
    #                 "hw3/models/1_layer_lstm_p2_modified_epoch_{}".format(_+2))

    test_data_raw = np.load("/home/kiriteegak/Desktop/github-general/"
                            "cmu-deep-learning-2018/hw3/data/dev.npy")
    test_data_labels = np.load("/home/kiriteegak/Desktop/github-general/"
                               "cmu-deep-learning-2018/hw3/data/dev_phonemes.npy")

    model = load_model("/home/kiriteegak/Desktop/github-general/cmu-deep-learning-2018/"
                       "hw3/models/1_layer_lstm_p2_modified_epoch_9")

    acc = []
    for d, labels in make_data(data_size=test_data_raw.shape[0],
                               train_data_raw=test_data_raw,
                               train_data_labels=test_data_labels):
        acc.append(np.mean((labels * model.predict(d)).sum(axis=2).sum(axis=1)/max_size))
    print("Average accuracy {}".format(np.mean(np.array(acc)) * 100))
