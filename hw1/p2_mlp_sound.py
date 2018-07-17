import numpy as np

from keras import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy


def prepare_training_data(train_data_, training_labels_, slice_index=None, max_slice=None):
    X = np.concatenate(list(train_data_))
    Y_ = np.concatenate(list(training_labels_))
    if slice_index is not None and max_slice:
        Y_ = Y_[slice_index:slice_index + max_slice]
        X = X[slice_index:slice_index + max_slice]
    return X, to_categorical(Y_, num_classes=138)


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
        for _ in range(layers-1):
            model.add(Dense(dimensions[_+1], input_dim=40))
            model.add(Activation('relu'))
            model.add(Dropout(0.3))
        model.add(Dense(138))
        model.add(Activation('softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=[categorical_accuracy])

    model.fit(x, y, batch_size=batch_size, epochs=epochs)
    return model


def main():
    train_data = np.load("../resources/HW1P2/train.npy", encoding='bytes')
    training_labels = np.load("../resources/HW1P2/train_labels.npy", encoding='bytes')
    x_test = np.load("../resources/HW1P2/dev.npy", encoding='bytes')
    test_labels = np.load("../resources/HW1P2/dev_labels.npy", encoding='bytes')
    x_test, y_test = prepare_training_data(x_test, test_labels)

    model = None
    max_slice = 1000000

    n_epochs = 10

    for i in range(n_epochs):
        for _ in range(0, 15449191, max_slice):
            x_train, y_train = prepare_training_data(train_data,
                                                     training_labels,
                                                     slice_index=_,
                                                     max_slice=max_slice)
            model = make_model(x_train,
                               y_train,
                               layers=4,
                               dimensions=(1000, 1000),
                               batch_size=10,
                               epochs=1,
                               model=model)

            p = model.predict(x_test)
            p = (p == p.max(axis=1)[:, None]).astype(int)
            print("Accuracy is: {}".format(np.count_nonzero(p * y_test) * 100 / y_test.shape[0]))

    model.save("models/hw1p2_phoneme_prediction.h5py")
    model = load_model("models/hw1p2_phoneme_prediction.h5py")

    p = model.predict(x_test)
    p = (p == p.max(axis=1)[:, None]).astype(int)
    print("Accuracy is: {}".format(np.count_nonzero(p * y_test) * 100 / y_test.shape[0]))


main()
