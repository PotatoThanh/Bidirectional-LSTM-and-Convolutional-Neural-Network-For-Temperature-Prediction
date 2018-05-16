import numpy as np
import tensorflow as tf
import keras
from keras.layers import *
from keras.models import Model

def normal_model(input_shape):

    inputs = Input(input_shape)
    inputs = Flatten()(inputs)

    net = Dense(units=50, activation='relu')(inputs)
    net = Dropout(0.3)(net)
    net = Dense(units=30, activation='relu')(net)
    net = Dropout(0.3)(net)
    net = Dense(units=10, activation='relu')(net)
    net = Dropout(0.2)(net)
    net = Dense(units=5, activation='relu')(net)
    net = Dropout(0.1)(net)

    outputs = Dense(units=1)(net)

    return Model(inputs=inputs, outputs=outputs)

def LSTM_model(input_shape):

    with tf.name_scope('input'):
        inputs = Input(input_shape)

    with tf.name_scope('LSTM'):
        net = LSTM(100)(inputs)
        net = Dropout(0.3)(net)

    with tf.name_scope('Dense_100'):
        net = Dense(100, activation='relu')(net)
        net = Dropout(0.3)(net)

    with tf.name_scope('Dense_50'):
        net = Dense(50, activation='relu')(net)
        net = Dropout(0.2)(net)

    with tf.name_scope('Dense_10'):
        net = Dense(10, activation='relu')(net)
        net = Dropout(0.1)(net)

    with tf.name_scope('output'):
        outputs = Dense(1)(net)

    return Model(inputs=inputs, outputs=outputs)

def BLSTM_model(input_shape):

    with tf.name_scope('input'):
        inputs = Input(input_shape)

    with tf.name_scope('Conv1D'):
        conv1 = Conv1D(64, 2, padding='same')(inputs)
        conv1 = AveragePooling1D(2)(conv1)

    with tf.name_scope('BLSTM_forward'):
        net1 = Bidirectional(LSTM(80, go_backwards=False))(conv1)
        net1 = Dropout(0.3)(net1)

    with tf.name_scope('BLSTM_backward'):
        net2 = Bidirectional(LSTM(80, go_backwards=True))(conv1)
        net2 = Dropout(0.3)(net2)

    with tf.name_scope('Dense_100'):
        net = concatenate([net1, net2])
        net = Dense(100, activation='relu')(net)
        net = Dropout(0.3)(net)

    with tf.name_scope('Dense_50'):
        net = Dense(70, activation='relu')(net)
        net = Dropout(0.2)(net)

    with tf.name_scope('Dense_10'):
        net = Dense(20, activation='relu')(net)
        net = Dropout(0.1)(net)

    with tf.name_scope('output'):
        outputs = Dense(1)(net)

    return Model(inputs=inputs, outputs=outputs)
