import numpy as np
import pickle
import joblib

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Dropout, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Lambda
from keras.optimizers import Adam, SGD
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy

sgd_ = SGD(momentum=0.001, nesterov=True)


def modify_and_dump_data(data=None):
    if not data:
        TRAIN_DATA_RAW = np.load("/home/kiriteegak/Desktop/github-general/"
                                 "cmu-deep-learning-2018/hw2/data/p2/train-features.npy")[:300]
        TRAIN_DATA_LABELS = to_categorical(np.concatenate(np.load("/home/kiriteegak/Desktop/github-general/"
                                                                  "cmu-deep-learning-2018/hw2/data/p2/"
                                                                  "train-labels.npy")))[:300]
    else:
        TRAIN_DATA_RAW = np.load("/home/kiriteegak/Desktop/github-general/cmu-deep-learning-2018/"
                                 "hw2/data/dev-features.npy")
        TRAIN_DATA_LABELS = np.load("/home/kiriteegak/Desktop/github-general/cmu-deep-learning-2018/"
                                    "hw2/data/p2/train-labels.npy")

    l = []
    max_size = max([_[0].shape[0] for _ in TRAIN_DATA_RAW])
    print(max_size)

    l = [bucketize_points(_[0], to_one=max_size) for _ in TRAIN_DATA_RAW]
    # for _ in TRAIN_DATA_RAW:
    #     if not l:
    #         l.append(neighbors(_))
    #     else:
    #         l[0] += neighbors(_)

    if not data:
        joblib.dump(l, "data/p2/padded_unsliced_frames_joblib")
    else:
        with open("data/p2/sliced_frames_test.pkl", "wb") as f:
            pickle.dump(l[0], f)


def neighbors(x, max_shape):
    slices = [[x[1][_], x[1][_ + 1]] for _ in range(len(x[1]) - 1)]
    slices.append([x[1][len(x) - 2], -1])
    return [bucketize_points(x[0][i[0]: i[1]], bucket_size=10, to_one=max_shape) for i in slices]


def bucketize_points(ele, bucket_size=100, to_one=None):
    if not to_one:
        if ele.shape[0] % bucket_size:
            if not ele.shape[0] % 2:
                pad_ = int((((int(ele.shape[0] / bucket_size) + 1) * bucket_size) - ele.shape[0]) / 2)
                if pad_ < 1:
                    pad_1, pad_2 = 1, 0
                else:
                    pad_1, pad_2 = pad_, pad_
                return np.pad(ele, ((pad_1, pad_2), (0, 0)), 'constant', constant_values=(0,))
            else:
                pad_1 = int((((int(ele.shape[0] / bucket_size) + 1) * bucket_size) - ele.shape[0]) / 2)
                pad_2 = pad_1 + 1
                return np.pad(ele, ((pad_1, pad_2), (0, 0)), 'constant', constant_values=(0,))
        else:
            return ele
    else:
        if not (to_one - ele.shape[0]) % 2:
            pad_ = int(((to_one - ele.shape[0]) / 2))
            if pad_ < 1:
                pad_1, pad_2 = 1, 0
            else:
                pad_1, pad_2 = pad_, pad_
            return np.pad(ele, ((pad_1, pad_2), (0, 0)), 'constant', constant_values=(0,))
        else:
            pad_1 = int((to_one - ele.shape[0]) / 2)
            pad_2 = pad_1 + 1
            return np.pad(ele, [(pad_1, pad_2), (0, 0)], 'constant', constant_values=(0,))


def time_based_slicing(x, crop_at):
    dim = x.get_shape()
    len_ = crop_at[1] - crop_at[0]
    return tf.slice(x, [0, crop_at[0], 0, 0], [1, len_, dim[2], dim[3]])


def return_out_shape(input_shape):
    return tuple([input_shape[0], None, input_shape[2], input_shape[3]])


def cnn_architecture():
    model = Sequential()

    model.add(Conv2D(filters=48,
                     input_shape=(1, 1479, 40),
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
    model.add(Lambda(time_based_slicing, output_shape=return_out_shape, arguments={'crop_at': (2, 5)}))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=100, init='normal'))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))

    model.add(Dense(units=46, init='normal'))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(optimizer=sgd_, loss=categorical_crossentropy, metrics=['accuracy'])

    for epochs in range(5):
        print("\n>>>>>>\nEPOCH-{}\n>>>>>>>\n".format(epochs))
        for i, x in enumerate(points):
            x = np.array([[x]])
            model.fit(x, np.array([TRAIN_DATA_LABELS[i]]))
    model.save("models/phoneme_prediction_p2.h5py")


modify_and_dump_data()
points = joblib.load("data/p2/padded_unsliced_frames_joblib")


# with open("data/p2/sliced_frames.pkl", "rb") as f:
#     points = pickle.load(f)
# points = points[:3000]


with open("data/p2/one_hot_encoded_labels_p2.pkl", "rb") as f:
    TRAIN_DATA_LABELS = pickle.load(f)[:3000]

# c = list(zip(points, TRAIN_DATA_LABELS))
# random.shuffle(c)
# points, TRAIN_DATA_LABELS = zip(*c)


cnn_architecture()
