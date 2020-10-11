from copy import copy
import numpy as np
import h5py
import os

from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import math

def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps

def remove_incomplete_days(data, timestamps, T=48):
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i+T-1 < len(timestamps) and int(timestamps[i+T-1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps

def arima_prediction(data, T, len_test):
    train_data, test_data = data[:len_test], data[-len_test:]
    num_rows, num_columns = data.shape[2], data.shape[3]

    prediction_shape = (len_test, data.shape[1], data.shape[2], data.shape[3])
    predicted_data = np.empty(prediction_shape)

    for flow in [0,1]:
        for row in range(num_rows):
            for column in range(num_columns):
                history_region = [x[flow][row][column] for x in train_data]
                history_region = np.array(history_region)

                for i in range(len_test):
                    if (sum(history_region) == 0):
                        yhat = 0
                    else:
                        model = ARIMA(history_region, order=(1,0,0))
                        model_fit = model.fit(disp=0)
                        output = model_fit.forecast()
                        yhat = output[0]
                    predicted_data[i][flow][row][column] = yhat
                    obs = test_data[i][flow][row][column]
                    history_region = np.append(history_region, obs)

                print(f'flow {flow}, region {row}x{column}')
    
    return predicted_data

def evaluate(real_data, predicted_data):
    predicted_data_inflow = np.asarray([d[0].flatten() for d in predicted_data])
    predicted_data_outflow = np.asarray([d[1].flatten() for d in predicted_data])

    real_data_inflow = np.asarray([d[0].flatten() for d in real_data])
    real_data_outflow = np.asarray([d[1].flatten() for d in real_data])

    rmse_inflow = math.sqrt(mean_squared_error(real_data_inflow, predicted_data_inflow))
    rmse_outflow = math.sqrt(mean_squared_error(real_data_outflow, predicted_data_outflow))
    # mae = mean_absolute_error(real_data, data_predicted)
    return rmse_inflow, rmse_outflow

def arima_prediction_bikeNYC():
    DATAPATH = '../data'
    nb_flow = 2 # i.e. inflow and outflow
    T = 24 # number timestamps per day
    len_test = T * 10 # number of timestamps to predict (ten days)

    # load data
    fname = os.path.join(DATAPATH, 'BikeNYC', 'NYC14_M16x8_T60_NewEnd.h5')
    print("file name: ", fname)
    data, timestamps = load_stdata(fname)
    # print(timestamps)
    # remove a certain day which does not have 24 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
    data[data < 0] = 0.
    print('data shape: ' + str(data.shape))

    # make predictions
    predicted_data = arima_prediction(data, T, len_test)

    # evaluate
    real_data = data[-len_test:]
    rmse_inflow, rmse_outflow = evaluate(real_data, predicted_data)

    print('BikeNYC rmse inflow: {rmse1}\nBikeNYC rmse outflow: {rmse2}'.format(rmse1=rmse_inflow, rmse2=rmse_outflow))

def arima_prediction_taxiBJ():
    DATAPATH = '../data'
    nb_flow = 2 # i.e. inflow and outflow
    T = 48 # number timestamps per day
    len_test = T * 4 * 7 # number of timestamps to predict (four weeks)

    # load data
    data_all = []
    timestamps_all = list()
    for year in range(13, 17):
        fname = os.path.join(
            DATAPATH, 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
        print("file name: ", fname)
        data, timestamps = load_stdata(fname)
        # print(timestamps)
        # remove a certain day which does not have 48 timestamps
        data, timestamps = remove_incomplete_days(data, timestamps, T)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        data_all.append(data)
        timestamps_all.append(timestamps)
    timestamps_all = [timestamp for l in timestamps_all for timestamp in l]
    data_all = np.vstack(copy(data_all))
    print('data shape: ' + str(data_all.shape))

    # make predictions
    predicted_data = arima_prediction(data_all, T, len_test)

    # evaluate
    real_data = data_all[-len_test:]
    rmse_inflow, rmse_outflow = evaluate(real_data, predicted_data)

    print('TaxiBJ rmse inflow: {rmse1}\nTaxiBJ rmse outflow: {rmse2}'.format(rmse1=rmse_inflow, rmse2=rmse_outflow))

if __name__ == '__main__':
    # arima_prediction_taxiBJ()
    arima_prediction_bikeNYC()