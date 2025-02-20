import numpy as np

import tensorflow as tf

import keras.backend as K
from keras.models import Model
from keras.layers import LSTM, TimeDistributed, Dense, Input, Lambda
from keras.layers.core import Activation
from keras.optimizers import SGD

from keras.models import load_model


def lstm_architecture():
    input_sequence = Input(shape=(None, 40), name="input_sequence", dtype='float32')
    input_labels = Input(shape=(None,), name="input_labels")
    input_length = Input(shape=[1], name='input_length', dtype='int64')
    label_length = Input(shape=[1], name='label_length', dtype='int64')

    lstm_layer_1 = LSTM(256, return_sequences=True, activation='tanh', dropout=0.2)(input_sequence)
    lstm_layer_2 = LSTM(128, return_sequences=True, activation='tanh', dropout=0.2)(lstm_layer_1)
    time_distributed_dense_1 = TimeDistributed(Dense(139), name='time_dist_1')(lstm_layer_2)
    predictions = Activation(activation='softmax', name='softmax_layer')(time_distributed_dense_1)

    # Transposing is already done in ctc_batch_loss
    # predictions_reshaped = Lambda(lambda a: tf.transpose(a, [1, 0, 2]))(predictions)

    loss_ = Lambda(ctc_lambda_function, output_shape=(1,), name='ctc_loss')([predictions,
                                                                             input_labels,
                                                                             input_length,
                                                                             label_length])
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=0.5)
    model_ = Model(inputs=[input_sequence, input_labels, input_length, label_length], outputs=loss_)
    model_.compile(optimizer=sgd, loss={'ctc_loss': lambda y_true, y_pred: y_pred})
    return model_


def ctc_lambda_function(args):
    y_pred, labels_, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN tend to be garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels_, y_pred, input_length, label_length, ignore_longer_outputs_than_inputs=True)


def make_data(data_size_=None, train_data_raw_=None, train_data_labels_=None, batch_size=1, reshape_=None):
    for st in range(0, data_size_, batch_size):
        dict_ = {}
        padded_sequences, padded_labels, sequence_length, label_length = [], [], [], []
        for i, _ in enumerate(train_data_raw_[st:st + batch_size]):
            padded_sequences.append(_)
            padded_labels.append(train_data_labels_[i])
            sequence_length.append(_.shape[0] - 2)
            label_length.append(train_data_labels_[i].shape[0])
        if not reshape_:
            dict_['input_sequence'] = np.array(padded_sequences)
            dict_['input_labels'] = np.array(padded_labels)
            dict_['input_length'] = np.array(sequence_length).reshape((1, 1))
            dict_['label_length'] = np.array(label_length).reshape((1, 1))
        else:
            # How can i do this. Since I don't know the alignment. I cannot reshape the sequence.
            dict_['input_sequence'] = np.array(padded_sequences)
            dict_['input_labels'] = np.array(padded_labels)
            dict_['input_length'] = np.array(sequence_length)
            dict_['label_length'] = np.array(label_length)

        if not reshape_:
            yield dict_, {'ctc_loss': np.zeros(shape=(batch_size,))}
        else:
            # Has not implemented this. Because of reshaping and aligning
            # Question: How do I do this. Ask?
            yield np.array(padded_sequences).reshape((4, 683, 40)), \
                  np.array(padded_labels, dtype='float32').reshape((4, 683, 139))


def change_network_architecture(model_):
    model_.layers.pop()
    model_.layers.pop()
    model_.layers.pop()
    model_.layers.pop()
    model_.input.pop(-1)
    model_.input.pop(-1)
    model_.input.pop(-1)
    model_.summary()
    model_changed_output = Model(model_.inputs, model_.layers[-1].output)
    return model_changed_output


def training_():
    train_data_raw = np.load("/home/tatras/Desktop/"
                             "github-general/cmu-deep-learning-2018/"
                             "hw3/data/train.npy",
                             encoding="bytes")
    train_data_labels = np.load("/home/tatras/Desktop/"
                                "github-general/cmu-deep-learning-2018/"
                                "hw3/data/train_phonemes.npy",
                                encoding="bytes")
    data_size = train_data_labels.shape[0]
    model = lstm_architecture()
    for _ in range(3):
        model.fit_generator(make_data(data_size_=data_size,
                                      train_data_raw_=train_data_raw,
                                      train_data_labels_=train_data_labels,
                                      reshape_=False),
                            shuffle=False,
                            steps_per_epoch=data_size)
        model.save("/home/tatras/Desktop/github-general/cmu-deep-learning-2018/"
                   "hw3/models/2_layer_lstm_ctc_epoch_{}".format(_))


def testing_():
    # Training the data in generators
    test_data_raw = np.load("/home/kiriteegak/Desktop/github-general/"
                            "cmu-deep-learning-2018/hw3/data/dev.npy")
    sizes = np.apply_along_axis(len, 0, test_data_raw)
    test_data_raw = np.apply_along_axis(np.expand_dims, 0, test_data_raw, 1)
    model = load_model("/home/kiriteegak/Desktop/github-general/cmu-deep-learning-2018/"
                       "hw3/models/2_layer_lstm_ctc_epoch_0",
                       custom_objects={'tf': tf})
    print("here")
    model_changed = change_network_architecture(model)
    return model_changed.predict(x=test_data_raw), sizes


if __name__ == '__main__':
    test_data_labels = np.load("/home/kiriteegak/Desktop/github-general/"
                               "cmu-deep-learning-2018/hw3/data/dev_phonemes.npy")
    outputs, lengths_ = testing_()
    print(K.ctc_decode(outputs, lengths_, greedy=False))
