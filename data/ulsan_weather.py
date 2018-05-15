import csv
import config
from model.my_model import *

# Parameter
LEARNING_RATE = config.LEARNING_RATE
WINDOW_SIZE = config.WINDOW_SIZE
FEATURE_SIZE = config.FEATURE_SIZE

NUM_TRAIN = config.NUM_TRAIN
NUM_VAL = config.NUM_VAL

def get_data(file_path):
    # IMPORT DATA
    data = []
    with open(file_path,'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = '\t')
        flag_1st = True
        for row in spamreader:
            if flag_1st :
                flag_1st = False
                continue
            eachrow = str(row[0])
            eachrow = eachrow.split(',')
            eachrow = np.array(eachrow[1:]).astype(np.float)
            data.append(eachrow)

    # data table
    data = np.array(data)

    # normalize temperature
    temper = data[:, 0]

    temper_min = np.min(temper)
    temper_max = np.max(temper)
    temper = (temper - temper_min)/(temper_max-temper_min)
    data[:, 0] = temper

    # normalize precipitation
    precipt = data[:, 1]

    precipt_min = np.min(precipt)
    precipt_max = np.max(precipt)
    precipt = (precipt-precipt_min)/(precipt_max-precipt_min)
    data[:, 1] = precipt

    # nomalize wind_speed
    wind_sp = data[:, 2]

    wind_sp_min = np.min(wind_sp)
    wind_sp_max = np.max(wind_sp)
    wind_sp = (wind_sp-wind_sp_min)/(wind_sp_max-wind_sp_min)
    data[:, 2] = wind_sp

    # nomalize wind_direction
    wind_dr = data[:, 3]

    wind_dr_min = np.min(wind_dr)
    wind_dr_max = np.max(wind_dr)
    wind_dr = (wind_dr-wind_dr_min)/(wind_dr_max-wind_dr_min)
    data[:, 3] = wind_dr

    # nomalize humid
    humid = data[:, 4]

    humid_min = np.min(humid)
    humid_max = np.max(humid)
    humid = (humid-humid_min)/(humid_max-humid_min)
    data[:, 4] = humid

    # nomalize sea level
    sea_lv = data[:, 5]

    sea_lv_min = np.min(sea_lv)
    sea_lv_max = np.max(sea_lv)
    sea_lv = (sea_lv-sea_lv_min)/(sea_lv_max-sea_lv_min)
    data[:, 5] = sea_lv

    # data generator
    features = []
    predict = []
    for i in range(len(data) - WINDOW_SIZE):
        x = np.array(data[i:i + WINDOW_SIZE, :]).flatten()
        y = data[i+WINDOW_SIZE, 0]

        features.append(x)
        predict.append(y)

    features = np.array(features)
    predict = np.array(predict)

    # train samples
    x_train = np.reshape(features[0:NUM_TRAIN, :], (NUM_TRAIN, WINDOW_SIZE, FEATURE_SIZE))
    y_train = np.reshape(predict[0:NUM_TRAIN], (NUM_TRAIN, -1))

    # validation samples
    x_val = np.reshape(features[NUM_TRAIN: NUM_TRAIN+NUM_VAL, :], (NUM_VAL, WINDOW_SIZE, FEATURE_SIZE))
    y_val = np.reshape(predict[NUM_TRAIN: NUM_TRAIN+NUM_VAL], (NUM_VAL, -1))

    # test samples
    NUM_TEST = len(predict) - NUM_TRAIN - NUM_VAL
    x_test = np.reshape(features[NUM_TRAIN+NUM_VAL: len(predict), :], (NUM_TEST, WINDOW_SIZE, FEATURE_SIZE))
    y_test = np.reshape(predict[NUM_TRAIN+NUM_VAL: len(predict)], (NUM_TEST, -1))

    return (x_train, y_train, x_val, y_val, y_test, x_test, temper_min, temper_max, NUM_TEST)