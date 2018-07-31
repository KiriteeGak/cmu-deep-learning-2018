import numpy as np
import pickle
import joblib

import tensorflow as tf

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Activation, Dropout, AveragePooling2D, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Lambda
from keras.optimizers import Adam, SGD
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy

sgd_ = SGD(momentum=0.001, nesterov=True)


def modify_and_dump_data(data=None):
    if not data:
        TRAIN_DATA_RAW = np.load("/home/kiriteegak/Desktop/github-general/"
                                 "cmu-deep-learning-2018/hw2/data/p2/train-features.npy")[:1]
        TRAIN_DATA_LABELS = to_categorical(np.concatenate(np.load("/home/kiriteegak/Desktop/github-general/"
                                                                  "cmu-deep-learning-2018/hw2/data/p2/"
                                                                  "train-labels.npy")[:1]))
    else:
        TRAIN_DATA_RAW = np.load("/home/kiriteegak/Desktop/github-general/cmu-deep-learning-2018/"
                                 "hw2/data/dev-features.npy")
        TRAIN_DATA_LABELS = np.load("/home/kiriteegak/Desktop/github-general/cmu-deep-learning-2018/"
                                    "hw2/data/p2/train-labels.npy")

    max_size = max([_[0].shape[0] for _ in TRAIN_DATA_RAW])

    padded_data_array, masked_array = [], []

    for _ in TRAIN_DATA_RAW:
        padded_data, mask_created = bucketize_points(_, to_one=max_size)
        padded_data_array.append(padded_data)
        masked_array.append(mask_created)
    padded_data_array = np.array(padded_data_array)
    masked_array = np.array(masked_array)

    # joblib.dump(np.array(slice_ranges), "/home/kiriteegak/Desktop/github-general"
    #                                     "/cmu-deep-learning-2018/"
    #                                     "hw2/data/p2/slices_ranges")
    # if not data:
    #     joblib.dump(padded_data, "/home/kiriteegak/Desktop/github-general/cmu-deep-learning-2018/"
    #                              "hw2/data/p2/padded_unsliced_frames_joblib")
    # else:
    #     with open("data/p2/sliced_frames_test.pkl", "wb") as f:
    #         pickle.dump(padded_data[0], f)


def neighbors(x, max_shape):
    slices = [[x[1][_], x[1][_ + 1]] for _ in range(len(x[1]) - 1)]
    slices.append([x[1][len(x) - 2], -1])
    return [bucketize_points(x[0][i[0]: i[1]], bucket_size=10, to_one=max_shape) for i in slices]


def slice_ranges_each_recording(x, max_shape, pad_1, max_len):
    slices = [[x[_], x[_ + 1]] for _ in range(len(x) - 1)]
    slices.append([x[len(x) - 1], max_len])
    masks = []
    for _s in slices:
        zeros_mask = np.zeros(shape=(max_shape, 40))
        one_mask = np.ones(shape=(_s[1] - _s[0], 40))
        zeros_mask[pad_1 + _s[0]: pad_1 + _s[1]] = one_mask
        masks.append(zeros_mask)
    return masks


def bucketize_points(ele_, bucket_size=100, to_one=None):
    ele, slice_r = ele_

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
            pad_ = (to_one - ele.shape[0]) / 2.0
            if not pad_:
                pad_1, pad_2 = 0, 0
            elif pad_ == 0.5:
                pad_1, pad_2 = 1, 0
            else:
                pad_1, pad_2 = int(pad_), int(pad_)
            ele = np.pad(ele, ((pad_1, pad_2), (0, 0)), 'constant', constant_values=(0,))
        else:
            pad_1 = int((to_one - ele.shape[0]) / 2)
            pad_2 = pad_1 + 1
            ele = np.pad(ele, [(pad_1, pad_2), (0, 0)], 'constant', constant_values=(0,))

        # Generating each slice masks
        slices = np.array(slice_ranges_each_recording(slice_r, to_one, pad_1=pad_1, max_len=ele_[0].shape[0]))
        return ele, slices


def time_based_slicing(x):
    x, crop_at = x[0], x[1]
    dim = x.get_shape()
    x = tf.squeeze(x)
    return tf.slice(x, [0, 2, 0, 0], [1, 3, dim[2], dim[3]])


def return_out_shape(input_shape):
    return tuple([input_shape[0], None, input_shape[2], input_shape[3]])


def cnn_architecture(X, slice_tensors, train_data_labels):
    main_input = Input(shape=(1, 1479, 40), dtype='float32', name='main_input')
    custom_pooling = Input(shape=(2,), dtype='float16', name='custom_pooling')

    conv1 = Conv2D(filters=48,
                   border_mode='same',
                   kernel_size=3,
                   use_bias=True,
                   data_format="channels_last",
                   activation='relu')(main_input)
    conv2 = Conv2D(filters=48,
                   border_mode='same',
                   kernel_size=2,
                   use_bias=True,
                   activation='relu')(conv1)
    pool1 = AveragePooling2D(pool_size=(1, 2))(conv2)
    conv3 = Conv2D(filters=96,
                   kernel_size=(2, 2),
                   border_mode='same',
                   activation='relu')(pool1)
    conv4 = Conv2D(filters=96,
                   kernel_size=(2, 2),
                   border_mode='same',
                   activation='relu')(conv3)
    pool2 = Lambda(time_based_slicing)([conv4, custom_pooling])
    global_pool = GlobalAveragePooling2D()(pool2)
    dense1 = Dense(units=100, init='normal', activation='relu')(global_pool)
    dense2 = Dense(units=46, init='normal', activation='softmax')(dense1)

    model = Model(inputs=[main_input, custom_pooling], outputs=[dense2])
    model.compile(optimizer=sgd_, loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()


if __name__ == '__main__':
    modify_and_dump_data()
    # points = joblib.load("/home/kiriteegak/Desktop/github-general/cmu-deep-learning-2018/"
    #                      "hw2/data/p2/padded_unsliced_frames_joblib")
    # ranges = joblib.load("/home/kiriteegak/Desktop/github-general/cmu-deep-learning-2018/"
    #                      "hw2/data/p2/slices_ranges")
    # with open("/home/kiriteegak/Desktop/github-general/cmu-deep-learning-2018/hw2/"
    #           "data/p2/one_hot_encoded_labels_p2.pkl", "rb") as f:
    #     TRAIN_DATA_LABELS = pickle.load(f)[:22507]

    # c = list(zip(points, TRAIN_DATA_LABELS))
    # random.shuffle(c)
    # points, TRAIN_DATA_LABELS = zip(*c)

    # cnn_architecture(X=points, slice_tensors=ranges, train_data_labels=TRAIN_DATA_LABELS)
