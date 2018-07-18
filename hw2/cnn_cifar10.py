from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Dense, Flatten, Activation

import numpy as np


class PreProcessing(object):
    def __init__(self):
        self.X, self.Y = self.load_data()

    @staticmethod
    def load_data():
        import os
        import pickle

        import numpy as np

        data = []
        labels = []

        files = [f for f in os.listdir("cifar-10-batches-py") if "data_batch" in f]

        for f_name in files:
            with open("cifar-10-batches-py/{}".format(f_name), 'rb') as fo:
                dict_ = pickle.load(fo, encoding='bytes')
                data.append(dict_[b'data'])
                labels.append(dict_[b'labels'])
        return np.concatenate(data), np.concatenate(labels)

    def sample_zero_mean(self):
        return np.apply_along_axis(lambda arr: (arr-arr.mean())/arr.std(), -1, self.X)

    def feature_zero_mean(self):
        for label_id in range(10):
            idx = np.where(self.Y == label_id)
            self.X[idx] = (self.X[idx] - self.X[idx].mean())/self.X[idx].std()

    def zca(self, x, s, lambda_, epsilon):
        x_average = np.mean(x)
        x = x - x_average
        contrast = np.sqrt(lambda_ + np.mean(x ** 2))
        return s * x / max(contrast, epsilon)

    def _reshape(self):
        return np.apply_along_axis(lambda a: np.reshape(a, (32, 32, 3)), -1, self.X)

    def chain_pre_processes(self):
        pass


def cnn_architecture():
    """
    Architecture
    3 × 3 conv. 96 ReLU
    3 × 3 conv. 96 ReLU
    3 × 3 conv. 96 ReLU with stride r = 2
    3 × 3 conv. 192 ReLU
    3 × 3 conv. 192 ReLU
    3 × 3 conv. 192 ReLU with stride r = 2
    :return:
    """

    model = Sequential()
    model.add(Conv2D(96, 3, 3, 3, input_shape=(3, 32, 32)))
    model.add(Activation('relu'))
    model.add(Conv2D(96, 96, 3, 3))
    model.add(Activation('relu'))
    model.add(Conv2D(96, 96, 3, 3, strides=2))
    model.add(Activation('relu'))
    model.add(Conv2D(192, 96, 3, 3))
    model.add(Activation('relu'))
    model.add(Conv2D(192, 192, 3, 3))
    model.add(Activation('relu'))
    model.add(Conv2D(192, 192, 3, 3, strides=2))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(192*8*8, 1000, init='normal'))
    model.add(Activation('relu'))
    model.add(Dense(1000, 10, init='normal'))
    model.add(Activation('softmax'))
