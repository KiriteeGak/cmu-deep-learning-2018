import numpy as np

from keras import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy


def prepare_training_data(train_data_, training_labels_, slice_index=None, max_slice=None, stride=2, index_=2):
    for _ in range(0, len(train_data_), 2000):
        train_data__ = train_data_[_:_ + 2000]
        training_labels__ = training_labels_[_:_ + 2000]

        if not index_ == 0 and not stride == 0:
            X = np.concatenate([_with_index_and_stride(e, index_=index_, stride=stride)
                                for e in train_data__])

            Y_ = np.concatenate(training_labels__)
        else:
            X = np.concatenate(list(train_data__))
            Y_ = np.concatenate(list(training_labels__))

        if slice_index is not None and max_slice:
            idx = np.random.permutation(len(X))
            X, Y_ = X[idx], Y_[idx]
            Y_ = Y_[slice_index:slice_index + max_slice]
            X = X[slice_index:slice_index + max_slice]
        yield X, to_categorical(Y_, num_classes=138)


def _with_index_and_stride(x, index_, stride):
    y = np.zeros((x.shape[0] + 2 * stride, x.shape[1]))
    y[stride: y.shape[0] - stride] = x
    return np.array([np.concatenate(y[_ - stride: _ + stride + 1]) for _ in range(index_, np.shape(y)[0] - stride)])


def make_one_hot_vector(index_, max_length=138):
    base_one_hot = np.zeros(max_length)
    base_one_hot[index_] = 1
    return base_one_hot


def make_model(x, y, layers=1, dimensions=(200,), epochs=2, batch_size=32, model=None):
    if not model:
        model = Sequential()
        model.add(Dense(dimensions[0], input_dim=40))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        for _ in range(layers - 1):
            model.add(Dense(dimensions[_ + 1], input_dim=40))
            model.add(Activation('relu'))
            model.add(Dropout(0.3))
        model.add(Dense(138))
        model.add(Activation('softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=[categorical_accuracy])

    model.fit(x, y, batch_size=batch_size, epochs=epochs)
    return model


def main():
    max_slice = 1000000

    train_data = np.load("../resources/HW1P2/train.npy", encoding='bytes')
    training_labels = np.load("../resources/HW1P2/train_labels.npy", encoding='bytes')
    x_test = np.load("../resources/HW1P2/dev.npy", encoding='bytes')
    test_labels = np.load("../resources/HW1P2/dev_labels.npy", encoding='bytes')

    model = load_model("models/hw1p2_phoneme_prediction_1000.h5py")

    for x_test, y_test in prepare_training_data(x_test, test_labels, slice_index=None,
                                                max_slice=max_slice,
                                                index_=1,
                                                stride=1):
        p = model.predict(x_test)
        p = (p == p.max(axis=1)[:, None]).astype(int)
        print("Accuracy is: {}".format(np.count_nonzero(p * y_test) * 100 / y_test.shape[0]))


# n_epochs = 10

# for i in range(n_epochs):
# for x_train, y_train in prepare_training_data(train_data,
#                                               training_labels,
#                                               slice_index=None,
#                                               max_slice=max_slice,
#                                               index_=2,
#                                               stride=2):


#         model = make_model(x_train,
#                            y_train,
#                            layers=4,
#                            dimensions=(1000, 100),
#                            batch_size=32,
#                            epochs=1,
#                            model=model)
#         p = model.predict(x_test)
#         p = (p == p.max(axis=1)[:, None]).astype(int)
#         print("Accuracy is: {}".format(np.count_nonzero(p * y_test) * 100 / y_test.shape[0]))

# model.save("models/hw1p2_phoneme_prediction.h5py")

main()
