import numpy as np
import pickle
import joblib

import tensorflow as tf

import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Conv2D, AveragePooling2D, Input, Lambda, Flatten
from keras.layers import dot
from keras.optimizers import Adam, SGD
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy

sgd_ = SGD(momentum=0.001, nesterov=True)


def modify_and_dump_data():
    TRAIN_DATA_RAW = np.load("/home/kiriteegak/Desktop/github-general/"
                             "cmu-deep-learning-2018/hw2/data/p2/train-features.npy")[:2]
    TRAIN_DATA_LABELS = to_categorical(np.concatenate(np.load("/home/kiriteegak/Desktop/github-general/"
                                                              "cmu-deep-learning-2018/hw2/data/p2/"
                                                              "train-labels.npy")))[:2]
    max_size = max([_[0].shape[0] for _ in TRAIN_DATA_RAW])
    padded_data_array, masked_array = [], []
    for i in range(0, TRAIN_DATA_RAW.shape[0], 32):
        for _ in TRAIN_DATA_RAW[i: i+32]:
            padded_data, mask_created = bucketize_points(_, to_one=max_size)
            padded_data_array.append(_replicate_data_points(padded_data, replication=_[1].shape[0]))
            masked_array.append(mask_created)
        padded_data_array = np.expand_dims(np.concatenate(np.array(padded_data_array)), axis=0)
        masked_array = np.expand_dims(np.concatenate(np.array(masked_array)), axis=0)
        return padded_data_array, masked_array, TRAIN_DATA_LABELS


def _replicate_data_points(point, replication):
    return np.array([point for _ in range(replication)])


def neighbors(x, max_shape):
    slices = [[x[1][_], x[1][_ + 1]] for _ in range(len(x[1]) - 1)]
    slices.append([x[1][len(x) - 2], -1])
    return [bucketize_points(x[0][i[0]: i[1]], bucket_size=10, to_one=max_shape) for i in slices]


def slice_ranges_each_recording(x, max_shape, pad_1, max_len):
    slices = [[x[_], x[_ + 1]] for _ in range(len(x) - 1)]
    slices.append([x[len(x) - 1], max_len])
    masks = []
    for _s in slices:
        zeros_mask = np.zeros(shape=(max_shape, 1))
        one_mask = np.ones(shape=(_s[1] - _s[0], 1))
        zeros_mask[pad_1 + _s[0]: pad_1 + _s[1]] = one_mask
        masks.append(zeros_mask.T)
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


def cnn_architecture(X=None, slice_tensors=None, train_data_labels=None):
    # train_data_labels = np.expand_dims(train_data_labels, axis=0)

    def time_based_slicing(x):
        return tf.squeeze(tf.tensordot(x[1], x[0], axes=1), [1, 2]) / tf.reduce_sum(x[1])

    main_input = Input(shape=(1, 477, 40), dtype='float32', name='main_input')
    custom_pooling = Input(shape=(1, 477, 1), dtype='float32', name='custom_pooling')

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

    pool1 = AveragePooling2D(pool_size=(1, 1))(conv2)

    conv3 = Conv2D(filters=96,
                   kernel_size=(2, 2),
                   border_mode='same',
                   activation='relu')(pool1)
    conv4 = Conv2D(filters=96,
                   kernel_size=(2, 2),
                   border_mode='same',
                   activation='relu')(conv3)

    # No lambda layer needed, custom lambda function to convert things to keras tensor
    # pool2 = Lambda(time_based_slicing)([conv4, custom_pooling])
    pool2 = dot([custom_pooling, conv4], axes=2)
    keras_tensor_conv_1 = Lambda(lambda a: tf.squeeze(a, axis=[1, 2]))(pool2)

    # flatten_layer = Flatten()(keras_tensor_conv_1)

    dense1 = Dense(units=100, init='normal', activation='relu')(keras_tensor_conv_1)
    dense2 = Dense(units=46, init='normal', activation='softmax')(dense1)
    # keras_tensor_conv_2 = Lambda(lambda a: tf.squeeze(a, axis=[1]))(dense2)

    model = Model(inputs=[main_input, custom_pooling], outputs=dense2)
    model.compile(optimizer=sgd_, loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()

    model.fit(x=[np.expand_dims(np.expand_dims(X, 0), 1), np.expand_dims(np.expand_dims(slice_tensors, 0), 0)],
              y=[np.expand_dims(np.expand_dims(train_data_labels, axis=0), axis=0)],
              batch_size=1,
              shuffle=True,
              epochs=200)


def dual_generator():
    pass


if __name__ == '__main__':
    X, pads, y = modify_and_dump_data()
    cnn_architecture(X=X[0][0], slice_tensors=pads[0][0].T, train_data_labels=y[0])
