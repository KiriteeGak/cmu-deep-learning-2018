import numpy as np
from time import time

import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Conv2D, AveragePooling2D, Input, Lambda
from keras.layers import dot
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy

sgd_ = SGD(momentum=0.001, nesterov=True)

train_data_raw = np.load("/home/kiriteegak/Desktop/github-general/"
                         "cmu-deep-learning-2018/hw2/data/p2/train-features.npy")
train_data_labels = np.load("/home/kiriteegak/Desktop/github-general/"
                            "cmu-deep-learning-2018/hw2/data/p2/"
                            "train-labels.npy")
MAX_SIZE = max([_[0].shape[0] for _ in train_data_raw])


def modify_and_dump_data():
    for i, _ in enumerate(train_data_raw):
        padded_data_array, masked_array = [], []
        padded_data, mask_created = bucketize_points(_, to_one=MAX_SIZE)
        padded_data_array.append(_replicate_data_points(padded_data, replication=_[1].shape[0]))
        masked_array.append(mask_created)
        padded_data_array_ = np.concatenate(np.array(padded_data_array))
        masked_array_ = np.concatenate(np.array(masked_array))
        yield padded_data_array_, masked_array_, to_categorical(train_data_labels[i], num_classes=46)


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


def cnn_architecture():
    main_input = Input(shape=(1, MAX_SIZE, 40), dtype='float32', name='main_input')
    custom_pooling = Input(shape=(1, MAX_SIZE, 1), dtype='float32', name='custom_pooling')

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
    dense1 = Dense(units=100, init='normal', activation='relu')(keras_tensor_conv_1)
    dense2 = Dense(units=46, init='normal', activation='softmax')(dense1)

    model = Model(inputs=[main_input, custom_pooling], outputs=dense2)
    model.compile(optimizer=sgd_, loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    # modify_and_dump_data()
    cnn_model = cnn_architecture()
    st = time()
    for e in range(30):
        for X, pads, Y in modify_and_dump_data():
            for x, pad, y in zip(X, pads, Y):
                cnn_model.fit(x=[np.expand_dims(np.expand_dims(x, 0), 1),
                                 np.expand_dims(np.expand_dims(pad, 0), 0)],
                              y=[np.expand_dims(np.expand_dims(y, axis=0), axis=0)],
                              batch_size=1,
                              shuffle=True,
                              verbose=0)
        print("IN EPOCH {}; Time consumed {} seconds".format(e, time()-st))
        st = time()
    cnn_model.save("models/phoneme_batch_1_update")
