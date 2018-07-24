import numpy as np
import pickle

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv1D, Activation, Dropout, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy


# TRAIN_DATA_RAW = np.load("/home/kiriteegak/Desktop/github-general/"
#                          "cmu-deep-learning-2018/hw2/data/p2/train-features.npy")
TRAIN_DATA_LABELS = np.load("/home/kiriteegak/Desktop/github-general/"
                            "cmu-deep-learning-2018/hw2/data/p2/train-labels.npy")


def cnn_architecture(X, y):
    model = Sequential()

    model.add(Conv2D(filters=40, input_shape=(None, None, 1), border_mode='same', kernel_size=(3, 3), use_bias=True))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=40, kernel_size=(3, 3), border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(AveragePooling2D(pool_size=(1, 2)))

    model.add(Conv2D(filters=80, kernel_size=(2, 2), border_mode='same'))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=80, kernel_size=(2, 2), border_mode='same'))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=1000, init='normal'))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))

    model.add(Dense(units=138, init='normal'))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()

    model.fit(X, y, epochs=2)


def neighbors(x):
    slices = [[x[1][_], x[1][_]+1] for _ in range(len(x[1])-1)]
    return [x[0][i[0]: i[1]] for i in slices]


# l = []
# for _ in TRAIN_DATA_RAW:
#     if not l:
#         l.append(neighbors(_))
#     else:
#         l[0] += neighbors(_)
#
#
# with open("data/p2/sliced_frames.pkl", "wb") as f:
#     pickle.dump(l, f)


with open("data/p2/sliced_frames.pkl", "rb") as f:
    points = pickle.load(f)
