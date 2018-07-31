from keras.layers import Dense, Lambda, Input
from keras.models import Model
import numpy as np


import tensorflow as tf


def output_testing(x):
    return tf.multiply(x[0], x[1])


main_input = Input(shape=(20, 10))
aux_input = Input(shape=(20, 1))

input1 = Dense(units=10, activation='relu')(main_input)
mul_vector = Lambda(output_testing)([input1, aux_input])
input2 = Dense(units=2, activation='softmax')(mul_vector)

X, aux, y = np.random.rand(1, 20, 10), np.random.rand(1, 20, 1), np.random.rand(1, 20, 2)

model = Model(inputs=[main_input, aux_input], outputs=[input2])

model.compile(optimizer='rmsprop', loss="mean_squared_error")

model.summary()

model.fit(x=[X, aux], y=[y], epochs=30, batch_size=1)
