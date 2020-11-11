import h5py
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
from matplotlib import pyplot
from keras import backend as K

def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps

def remove_incomplete_days(data, timestamps, T=48, h0_23=False):
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    first_timestamp_index = 0 if h0_23 else 1
    last_timestamp_index = T-1 if h0_23 else T
    while i < len(timestamps):
        if int(timestamps[i][8:]) != first_timestamp_index:
            i += 1
        elif i+T-1 < len(timestamps) and int(timestamps[i+T-1][8:]) == last_timestamp_index:
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

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def evaluate(real_data, predicted_data):
    # predicted_data_inflow = np.asarray([d[0].flatten() for d in predicted_data])
    # predicted_data_outflow = np.asarray([d[1].flatten() for d in predicted_data])

    # real_data_inflow = np.asarray([d[0].flatten() for d in real_data])
    # real_data_outflow = np.asarray([d[1].flatten() for d in real_data])

    # rmse_inflow = math.sqrt(mean_squared_error(real_data_inflow, predicted_data_inflow))
    # rmse_outflow = math.sqrt(mean_squared_error(real_data_outflow, predicted_data_outflow))
    # mae = mean_absolute_error(real_data, data_predicted)

    predicted_data_inflow = np.asarray([d[0] for d in predicted_data])
    predicted_data_outflow = np.asarray([d[1] for d in predicted_data])

    real_data_inflow = np.asarray([d[0] for d in real_data])
    real_data_outflow = np.asarray([d[1] for d in real_data])
    
    rmse_inflow = rmse(real_data_inflow, predicted_data_inflow)
    rmse_outflow = rmse(real_data_outflow, predicted_data_outflow)
    return rmse_inflow, rmse_outflow

def plot_region_data(real_data, predicted_data, region, flow):
    # region deve essere una lista o tupla di 2 elementi
    # flow deve essere 0 (inflow) o 1 (outflow)
    row, column = region[0], region[1]

    real_data_region = [x[flow][row][column] for x in real_data]
    predicted_data_region = [x[flow][row][column] for x in predicted_data]

    pyplot.plot(real_data_region)
    pyplot.plot(predicted_data_region, color='red')
    pyplot.legend(['real', 'predicted'])
    pyplot.show()
