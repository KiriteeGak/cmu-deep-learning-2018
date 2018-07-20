from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Dense, Flatten, Activation
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy

import numpy as np
from scipy.ndimage import rotate


class PreProcessing(object):
    def __init__(self):
        self.X, self.Y = self.load_data()
        self.Y = to_categorical(self.Y, num_classes=10)

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

    def feature_zero_mean(self, epsilon=10**-5):
        x_ = np.zeros(self.X.shape)
        _y = np.argmax(self.Y, axis=1)

        for label_id in range(10):
            idx = np.where(_y == label_id)
            mean_, std_ = self.X[idx].mean(), self.X[idx].std()
            x_[idx] = np.apply_along_axis(lambda a: (a-mean_)/(std_+epsilon), -1, self.X[idx])
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

    def shuffle_and_rotate(self):
        self.X = np.apply_along_axis(lambda a: a.reshape((32, 32, 3)), axis=-1, arr=self.X)
        idx = np.arange(self.X.shape[0])
        np.random.shuffle(idx)
        self.X, self.Y = self.X[idx], self.Y[idx]
        _X = [self.X]
        _Y = [self.Y]

        for angle in (90, 180, 270):
            _X.append(np.array(list(map(lambda a: rotate(a, angle=angle, axes=(0, 1)), self.X))))
            _Y.append(self.Y)
        self.X, self.Y = np.concatenate(_X), np.concatenate(_Y)

    def chain_pre_processes(self):
        self.sample_zero_mean()
        self.feature_zero_mean()
        self.global_contrast_normalisation()
        self.shuffle_and_rotate()
        return self.X, self.Y


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
    # X, Y = PreProcessing().chain_pre_processes()

    # np.save("data/train_data_transformed", X)
    # np.save("data/train_labels_transformed", Y)
    # exit()

    X = np.load("data/train_data_transformed.npy")
    Y = np.load("data/train_labels_transformed.npy")

    model = Sequential()
    model.add(Conv2D(filters=96,
                     kernel_size=3,
                     input_shape=(32, 32, 3),
                     border_mode='same',
                     data_format='channels_last'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=96, kernel_size=(3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=96, kernel_size=(3, 3), strides=2))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=192, kernel_size=(3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=192, kernel_size=(3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=2))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(units=1000, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1000, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, init='normal'))
    model.add(Activation('softmax'))

    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=[categorical_accuracy])
    model.fit(X, Y, epochs=10, batch_size=32)

    model.save("models/cifar_10_1000_1000_10_10.h5py")


# cnn_architecture()

