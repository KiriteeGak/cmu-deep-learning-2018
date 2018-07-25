import numpy as np
import pickle
import random

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv1D, Activation, Dropout, AveragePooling2D, Flatten
from keras import backend as K
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Lambda
from keras.optimizers import Adam, SGD
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy


sgd_ = SGD(momentum=0.001, nesterov=True)


def modify_and_dump_data(data=None):
    if not data:
        TRAIN_DATA_RAW = np.load("/home/kiriteegak/Desktop/github-general/"
                                 "cmu-deep-learning-2018/hw2/data/p2/train-features.npy")
        TRAIN_DATA_LABELS = to_categorical(np.concatenate(np.load("/home/kiriteegak/Desktop/github-general/"
                                                                  "cmu-deep-learning-2018/hw2/data/p2/"
                                                                  "train-labels.npy")))
    else:
        TRAIN_DATA_RAW= np.load("/home/kiriteegak/Desktop/github-general/cmu-deep-learning-2018/"
                                "hw2/data/dev-features.npy")
        TRAIN_DATA_LABELS = np.load("/home/kiriteegak/Desktop/github-general/cmu-deep-learning-2018/"
                                    "hw2/data/p2/train-labels.npy")
    l = []
    for _ in TRAIN_DATA_RAW:
        if not l:
            l.append(neighbors(_))
        else:
            l[0] += neighbors(_)
    if not data:
        with open("data/p2/sliced_frames.pkl", "wb") as f:
            pickle.dump(l[0], f)
    else:
        with open("data/p2/sliced_frames_test.pkl", "wb") as f:
            pickle.dump(l[0], f)


def cnn_architecture():
    model = Sequential()

    model.add(Conv2D(filters=48,
                     input_shape=(1, None, 40),
                     border_mode='same',
                     kernel_size=3,
                     use_bias=True,
                     data_format="channels_last"))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=48, kernel_size=(3, 3), border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(AveragePooling2D(pool_size=(1, 2)))

    model.add(Conv2D(filters=96, kernel_size=(2, 2), border_mode='same'))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=96, kernel_size=(2, 2), border_mode='same'))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))

    # Add a customised pooling layer
    # model.add(Lambda(time_based_pooling, output_shape=time_based_pooling_out_shape))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=100, init='normal'))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))

    model.add(Dense(units=46, init='normal'))
    model.add(Activation('softmax'))

    model.compile(optimizer=sgd_, loss=categorical_crossentropy, metrics=['accuracy'])

    for epochs in range(5):
        print("\n>>>>>>\nEPOCH-{}\n>>>>>>>\n".format(epochs))
        for i, x in enumerate(points):
            x = np.array([[x]])
            model.fit(x, np.array([TRAIN_DATA_LABELS[i]]))
    model.save("models/phoneme_prediction_p2.h5py")


def time_based_pooling_out_shape():
    return tuple((None, 80))


def neighbors(x):
    slices = [[x[1][_], x[1][_ + 1]] for _ in range(len(x[1]) - 1)]
    slices.append([x[1][len(x) - 2], -1])
    return [x[0][i[0]: i[1]] for i in slices]


# modify_and_dump_data()

with open("data/p2/sliced_frames.pkl", "rb") as f:
    points = pickle.load(f)


with open("data/p2/one_hot_encoded_labels_p2.pkl", "rb") as f:
    TRAIN_DATA_LABELS = pickle.load(f)


c = list(zip(points, TRAIN_DATA_LABELS))
random.shuffle(c)
points, TRAIN_DATA_LABELS = zip(*c)

# cnn_architecture()
