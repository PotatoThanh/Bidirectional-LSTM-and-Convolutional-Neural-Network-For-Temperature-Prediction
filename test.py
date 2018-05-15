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

if DELETE_TEST:
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

# load weights
ckpt_path = 'trained/checkpoint_/weights.1638-0.00.h5'
model.load_weights(ckpt_path)

# evaluate
MSE_loss = model.evaluate(x_test, y_test, verbose=1, batch_size=BATCH_SIZE)
print('MSE loss: ' + str(MSE_loss))

# predict
test_predict = model.predict(x_test, verbose=1, batch_size=BATCH_SIZE)

# reshape
test_predict = np.reshape(test_predict, (NUM_TEST, -1))
temper_truth = np.reshape(y_test, (NUM_TEST, -1))

# become original temperature
temper_predict = np.array(test_predict)*(temper_max - temper_min) + temper_min
temper_truth = np.array(temper_truth)*(temper_max - temper_min) + temper_min

# Error rate
temp = np.abs(temper_truth - temper_predict)
mean_loss = np.mean(temp)

print('Average loss: ' + str(mean_loss))
