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
        import numpy as np
        return np.apply_along_axis(lambda arr: (arr-arr.mean())/arr.std(), self.X)

    def feature_zero_mean(self):
        for label_id in range(10):
            idx = np.where(self.Y == label_id)
            self.X[idx] = (self.X[idx] - self.X[idx].mean())/self.X[idx].std()

    def zca(self, x, s, lambda_, epsilon):
        import numpy
        x_average = numpy.mean(x)
        x = x - x_average
        contrast = numpy.sqrt(lambda_ + numpy.mean(x ** 2))
        return s * x / max(contrast, epsilon)

    def chain_pre_processes(self):
        pass
