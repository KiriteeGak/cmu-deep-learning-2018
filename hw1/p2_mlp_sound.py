import numpy as np

from keras import Sequential
from keras.layers.core import Dense, Activation
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy


def prepare_training_data(train_data_, training_labels_):
    X = np.concatenate(list(train_data_))
    Y_ = np.concatenate(list(training_labels_))
    Y_ = Y_[:1000000]
    X = X[:1000000]
    return X, to_categorical(Y_, num_classes=138)


def make_one_hot_vector(index_, max_length=138):
    base_one_hot = np.zeros(max_length)
    base_one_hot[index_] = 1
    return base_one_hot


def make_model(X, y):
    model = Sequential()
    model.add(Dense(200, input_dim=40))
    model.add(Activation('relu'))
    model.add(Dense(138))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=[categorical_accuracy])
    model.fit(X, y, batch_size=32, epochs=1)
    return model


if __name__ == '__main__':
    train_data = np.load("../resources/HW1P2/train.npy", encoding='bytes')
    training_labels = np.load("../resources/HW1P2/train_labels.npy", encoding='bytes')
    x_train, y_train = prepare_training_data(train_data, training_labels)
    model = make_model(x_train, y_train)
    x_test = np.load("../resources/HW1P2/dev.npy", encoding='bytes')
    test_labels = np.load("../resources/HW1P2/dev_labels.npy", encoding='bytes')
    x_test, y_test = prepare_training_data(x_test, test_labels)
    p = model.predict(x_test)
    print(p)
