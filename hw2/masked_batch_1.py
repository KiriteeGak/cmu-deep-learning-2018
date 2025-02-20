import numpy as np
from time import time

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Dense, Conv2D, AveragePooling2D, Input, Lambda
from keras.layers import dot
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy

sgd_ = SGD(momentum=0.001, nesterov=True)

train_data_raw = np.load("/home/tatras/Desktop/github-general/"
                         "cmu-deep-learning-2018/hw2/data/p2/train-features.npy")
train_data_labels = np.load("/home/tatras/Desktop/github-general/"
                            "cmu-deep-learning-2018/hw2/data/p2/"
                            "train-labels.npy")
MAX_SIZE = max([_[0].shape[0] for _ in train_data_raw])


def modify_and_dump_data(data=None, labels=None):
    if data == None and labels == None:
        data = train_data_raw
        labels = train_data_labels

    for i, _ in enumerate(data):
        padded_data_array, masked_array = [], []
        padded_data, mask_created = bucketize_points(_, to_one=MAX_SIZE)
        padded_data_array.append(_replicate_data_points(padded_data, replication=_[1].shape[0]))
        masked_array.append(mask_created)
        padded_data_array_ = np.concatenate(np.array(padded_data_array))
        masked_array_ = np.concatenate(np.array(masked_array))
        yield padded_data_array_, masked_array_, to_categorical(labels[i], num_classes=46)


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
        masks.append(zeros_mask / zeros_mask.sum())
    return masks


def bucketize_points(ele_, bucket_size=100, to_one=None, masks=True, category=None):
    if masks and not category:
        ele, slice_r = ele_
    else:
        ele = ele_

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
        if masks:
            slices = np.array(slice_ranges_each_recording(slice_r, to_one, pad_1=pad_1, max_len=ele_[0].shape[0]))
            return ele, slices
        else:
            return create_epsilon_dimension(ele, pad_1, pad_2, to_categorical(category, num_classes=138))


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
    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    return model


def load_dev_and_predict(model_name):
    dev_data = "/home/tatras/Desktop/github-general/cmu-deep-learning-2018/hw2/data/p2/dev-features.npy"[:100]
    dev_labels = "/home/tatras/Desktop/github-general/cmu-deep-learning-2018/hw2/data/p2/dev-labels.npy"[:100]
    model = load_model(model_name, custom_objects={'tf': tf})
    total, correct = 0, 0
    for x_dev, pads_dev, y_dev in modify_and_dump_data(data=np.load(dev_data), labels=np.load(dev_labels)):
        for x, pad, y in zip(x_dev, pads_dev, y_dev):
            out = model.predict(x=[np.expand_dims(np.expand_dims(x, 0), 1),
                                   np.expand_dims(np.expand_dims(pad, 0), 0)])
            out = (out == out.max(axis=1)[:, None]).astype(int)
            total += 1
            correct += np.array([y]).dot(out.T)[0][0]
    print("The accuracy at the current time: ", correct * 100 / total)


def create_epsilon_dimension(ele, pad_1, pad_2, one_hots):
    epsilon_base_vector = np.zeros((ele.shape[0], 138))
    stack_vector = np.ones((ele.shape[0],))
    stack_vector[pad_1: ele.shape[0] - pad_2] = 0
    epsilon_base_vector[pad_1: pad_1+one_hots.shape[0], 0:138] = one_hots
    one_hots = epsilon_base_vector
    one_hots = np.hstack((one_hots, stack_vector.reshape((ele.shape[0], 1))))
    return ele, one_hots


if __name__ == '__main__':
    train_data_raw = np.load("/home/kiriteegak/Desktop/github-general/"
                             "cmu-deep-learning-2018/hw2/data/p2/train-features.npy")[:10]
    train_data_labels = np.load("/home/kiriteegak/Desktop/github-general/"
                                "cmu-deep-learning-2018/hw2/data/p2/"
                                "train-labels.npy")[:10]
    MAX_SIZE = max([_[0].shape[0] for _ in train_data_raw])

    cnn_model = cnn_architecture()

    # cnn_model = load_model("models/phoneme_batch_1_update_epoch_1", custom_objects={'tf': tf})

    st = time()
    for e in range(5):
        print("Started Epoch {}".format(e))
        for X, pads, Y in modify_and_dump_data():
            for x, pad, y in zip(X, pads, Y):
                cnn_model.fit(x=[np.expand_dims(np.expand_dims(x, 0), 1),
                                 np.expand_dims(np.expand_dims(pad, 0), 0)],
                              y=[np.expand_dims(np.expand_dims(y, axis=0), axis=0)],
                              batch_size=1,
                              shuffle=True,
                              verbose=0)
        cnn_model.save("models/phoneme_batch_1_update_epoch_{}".format(e), overwrite=True)
        load_dev_and_predict("models/phoneme_batch_1_update_epoch_{}".format(e))
        print("Completed Epoch {}; Time consumed {} seconds".format(e, time() - st))
        st = time()

    # load_dev_and_predict(model_name="models/phoneme_batch_1_update")
