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
        return np.apply_along_axis(lambda arr: (arr-arr.mean())/arr.std(), -1, self.X[:2])

    def feature_zero_mean(self):
        x_ = np.zeros(self.X.shape)
        for label_id in range(10):
            idx = np.where(self.Y == label_id)
            mean_, std_ = self.X[idx].mean(), self.X[idx].std()
            x_[idx] = np.apply_along_axis(lambda a: (a-mean_)/std_, -1, self.X[idx])
        self.X = x_

    # How is this going to be any more special than standard normalising ...
    def global_contrast_normalisation(self, s=1, lambda_=10**-5, epsilon=0.1):
        self.X = np.apply_along_axis(lambda a: s * (a-a.mean())/max(np.sqrt(lambda_ + np.mean(a ** 2)), epsilon),
                                     -1,
                                     self.X)

    def zca_normalisation(self):
        pass

    def _reshape(self):
        self.X = np.apply_along_axis(lambda a: np.reshape(a, (32, 32, 3)), -1, self.X)

    def chain_pre_processes(self):
        self.sample_zero_mean()
        self.feature_zero_mean()
        self.global_contrast_normalisation()


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
    model.add(Conv2D(96, 3, 3, 3, input_shape=(3, 32, 32), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, 96, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, 96, 3, 3, strides=2))
    model.add(Activation('relu'))
    model.add(Conv2D(192, 96, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, 192, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, 192, 3, 3, strides=2))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(192*8*8, 1000, init='normal'))
    model.add(Activation('relu'))
    model.add(Dense(1000, 10, init='normal'))
    model.add(Activation('softmax'))
