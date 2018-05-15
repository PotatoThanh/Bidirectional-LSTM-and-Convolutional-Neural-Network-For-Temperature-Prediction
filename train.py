import os

import config
from data.ulsan_weather import get_data
from model.my_model import *

# Parameter
DATA = config.DATA
LEARNING_RATE = config.LEARNING_RATE
WINDOW_SIZE = config.WINDOW_SIZE
FEATURE_SIZE = config.FEATURE_SIZE

NUM_TRAIN = config.NUM_TRAIN
NUM_VAL = config.NUM_VAL
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
DELETE_TRAIN = config.DELETE_TRAIN
DELETE_TEST = config.DELETE_TEST

if DELETE_TRAIN:
    path_logs = './logs/*'
    path_ckpt = './checkpoint/*'
    print('rm -Rf' + path_logs)
    os.system('rm -Rf ' + path_logs)

    print('rm -Rf' + path_ckpt)
    os.system('rm -Rf ' + path_ckpt)

# load data
x_train, y_train, x_val, y_val, y_test, x_test, temper_min, temper_max, NUM_TEST = get_data(DATA)

# print # samples
print('Number train samples: ' + str(NUM_TRAIN))
print('Number validation samples: ' + str(NUM_VAL))
print('Number test samples: ' + str(NUM_TEST))

# input shape
input_shape =(WINDOW_SIZE, FEATURE_SIZE)

# create model
model = BLSTM_model(input_shape)

# compile model with adam optimizer
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.mean_squared_error)

# call back functions
cb_tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=20, batch_size=BATCH_SIZE,
                                             write_graph=True, write_grads=True,
                                             write_images=False, embeddings_freq=0,
                                             embeddings_layer_names=None, embeddings_metadata=None)

cb_ckpt = keras.callbacks.ModelCheckpoint('./checkpoint/weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=1,
                                save_best_only=True, save_weights_only=False, mode='auto', period=10)
# train model
model.fit(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True,
          epochs=EPOCHS, validation_data=(x_val, y_val),
          callbacks=[cb_tensorboard, cb_ckpt])





