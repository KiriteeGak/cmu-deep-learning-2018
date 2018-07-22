from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dropout, Dense, Flatten, Activation, GlobalAveragePooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical

import numpy as np
from scipy.ndimage import rotate

sgd_ = SGD(momentum=0.9)


class PreProcessing(object):
    def __init__(self, folder_name, file_ending, rotate_=True, raw=False):
        self.folder_name = folder_name
        self.file_ending = file_ending
        self.rotate_ = rotate_
        self.raw = raw
        self.X, self.Y = self.load_data()
        self.Y = to_categorical(self.Y, num_classes=10)

    def load_data(self):
        import os
        import pickle

        data = []
        labels = []
        files = [f for f in os.listdir(self.folder_name) if self.file_ending in f]

        for f_name in files:
            with open("{}/{}".format(self.folder_name, f_name), 'rb') as fo:
                dict_ = pickle.load(fo, encoding='bytes')
                data.append(dict_[b'data'])
                labels.append(dict_[b'labels'])
        return np.concatenate(data), np.concatenate(labels)

    def sample_zero_mean(self):
        self.X = np.apply_along_axis(lambda arr: (arr - arr.mean()) / arr.std(), -1, self.X)

    def feature_zero_mean(self, epsilon=10 ** -5):
        x_ = np.zeros(self.X.shape)
        _y = np.argmax(self.Y, axis=1)

        for label_id in range(10):
            idx = np.where(_y == label_id)
            mean_, std_ = self.X[idx].mean(), self.X[idx].std()
            x_[idx] = np.apply_along_axis(lambda a: (a - mean_) / (std_ + epsilon), -1, self.X[idx])
        self.X = x_

    # How is this going to be any more special than standard normalising ...
    def global_contrast_normalisation(self, s=1, lambda_=10, epsilon=10 ** -6):
        self.X = np.apply_along_axis(lambda a: s * (a - a.mean()) / max(np.sqrt(lambda_ + np.mean(a ** 2)), epsilon),
                                     -1,
                                     self.X)

    # Complete zca normalisation.
    # Check https://stackoverflow.com/questions/31528800/how-to-implement-zca-whitening-python
    def zca_normalisation(self):
        """
        Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
        INPUT:  X: [M x N] matrix.
            Rows: Variables
            Columns: Observations
        Steps:
            Singular Value Decomposition. X = U * np.diag(S) * V
            U: [M x M] eigenvectors of sigma.
            S: [M x 1] eigenvalues of sigma.
            V: [M x M] transpose of U
            Whitening constant: prevents division by zero
            ZCA Whitening matrix: U * Lambda * U'
            Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
        """
        def zca_norm_helper(arr):
            sigma = np.dot(arr, arr.T)
            U, S, V = np.linalg.svd(np.array([[sigma]]))
            epsilon = 1e-5
            zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
            return np.dot(zca_matrix, arr.reshape((1, 3072)))
        self.X = self.X - self.X.mean()
        X_ = np.apply_along_axis(zca_norm_helper, axis=-1, arr=self.X)
        self.X = X_.reshape(50000, 3072)

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
        if not self.raw:
            self.sample_zero_mean()
            self.feature_zero_mean()
            self.global_contrast_normalisation()
            self.zca_normalisation()
            if self.rotate_:
                self.shuffle_and_rotate()
        self._reshape()
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
    3 × 3 conv. 192 ReLU
    1 × 1 conv. 192 ReLU
    1 × 1 conv. 10 ReLU
    global averaging over 6 × 6 spatial dimensions
    10 or 100-way softmax
    :return:
    """
    X, Y = PreProcessing(folder_name="cifar-10-batches-py",
                         file_ending="data_batch",
                         rotate_=False,
                         raw=True).chain_pre_processes()
    np.save("data/train_data_transformed_raw", X)
    np.save("data/train_labels_transformed_raw", Y)

    X = np.load("data/train_data_transformed_raw.npy")
    Y = np.load("data/train_labels_transformed_raw.npy")

    model = Sequential()
    model.add(Conv2D(filters=96,
                     kernel_size=3,
                     input_shape=(32, 32, 3),
                     border_mode='same',
                     data_format='channels_last',
                     kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=96, kernel_size=(3, 3), border_mode='same', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=96, kernel_size=(3, 3), strides=2, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=192, kernel_size=(3, 3), border_mode='same', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=192, kernel_size=(3, 3), border_mode='same', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=2, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=192, kernel_size=(3, 3), border_mode='same', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=192, kernel_size=(1, 1), border_mode='same', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=10, kernel_size=(1, 1), border_mode='same', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    # model.add(AveragePooling2D(pool_size=(6, 6)))
    model.add(GlobalAveragePooling2D())

    # model.add(Flatten())
    model.add(Dense(units=10, init='normal', kernel_regularizer=l2(0.001)))
    model.add(Activation('softmax'))

    model.compile(loss="categorical_crossentropy", optimizer=sgd_, metrics=["accuracy"])
    model.fit(X, Y, epochs=30, batch_size=32)
    model.save("models/cifar_10_1000_1000_10_1_regularizer_no_rotate.h5py")


# cnn_architecture()
model = load_model("models/cifar_10_1000_1000_10_1_regularizer_no_rotate.h5py")
x_test, y_test = PreProcessing(folder_name="cifar-10-batches-py",
                               file_ending="test_batch",
                               rotate_=False,
                               raw=True).chain_pre_processes()

print(np.count_nonzero(np.sum(model.predict(x_test)*y_test, axis=1)) * 100 / 10000)
