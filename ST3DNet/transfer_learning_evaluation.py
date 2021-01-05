from ST3DNet import *
import pickle
from utils import *
import os
import h5py
import math
import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from evaluation import evaluate

### 32x32
# params
# nb_epoch = 150  # number of epoch at training stage
# nb_epoch_cont = 200  # number of epoch at training (cont) stage
# batch_size = 32  # batch size
T = 24*4  # number of time intervals in one day
lr = 0.0001  # learning rate
# lr = 0.00002  # learning rate
len_closeness = 6  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 2  # length of trend dependent sequence
nb_residual_unit = 7   # number of residual units
nb_flow = 2  # there are two types of flows: new-flow and end-flow
days_test = 7  
len_test = T * days_test
map_height, map_width = 32, 32  # grid size
m_factor = 1

# load data
filename = os.path.join("../data", 'CACHE', 'ST3DNet', 'Rome_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))
f = open(filename, 'rb')
X_train = pickle.load(f)
Y_train = pickle.load(f)
X_test = pickle.load(f)
Y_test = pickle.load(f)
mmn = pickle.load(f)
external_dim = pickle.load(f)
timestamp_train = pickle.load(f)
timestamp_test = pickle.load(f)

for i in X_train:
    print(i.shape)

Y_train = mmn.inverse_transform(Y_train)  # X is MaxMinNormalized, Y is real value
Y_test = mmn.inverse_transform(Y_test)

c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
t_conf = (len_trend, nb_flow, map_height,
          map_width) if len_trend > 0 else None

# build model
model = ST3DNet(c_conf=c_conf, t_conf=t_conf, external_dim=external_dim, nb_residual_unit=nb_residual_unit)

# load weights
model_fname = 'TaxiBJ.c6.p0.t2.resunit7.lr0.0001.cont.noMeteo.best.h5'
model.load_weights(os.path.join('../best_models', 'ST3DNet', model_fname))

# predict
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'TL_taxiBJ_roma32x32_results.csv')
if not os.path.isfile(csv_name):
    if os.path.isdir('results') is False:
        os.mkdir('results')
    with open(csv_name, 'a', encoding = "utf-8") as file:
        file.write(
                'rsme_in,rsme_out,rsme_tot,'
                'mape_in,mape_out,mape_tot,'
                'ape_in,ape_out,ape_tot'
                )
        file.write("\n")
        file.close()
with open(csv_name, 'a', encoding = "utf-8") as file:
    file.write(f'{score[0]},{score[1]},{score[2]},{score[3]},'
            f'{score[4]},{score[5]},{score[6]},{score[7]},{score[8]}'
            )
    file.write("\n")
    file.close()

# save real vs predicted
fname = 'st3dnet_RomaNord32x32.h5'
h5 = h5py.File(fname, 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()

### 16x8
# params
# nb_epoch = 150  # number of epoch at training stage
# nb_epoch_cont = 200  # number of epoch at training (cont) stage
# batch_size = 32  # batch size
T = 24  # number of time intervals in one day
lr = 0.0001  # learning rate
# lr = 0.00002  # learning rate
len_closeness = 6  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 2  # length of trend dependent sequence
nb_residual_unit = 5   # number of residual units
nb_flow = 2  # there are two types of flows: new-flow and end-flow
days_test = 7  
len_test = T * days_test
map_height, map_width = 16, 8  # grid size
m_factor = 1

# load data
filename = os.path.join("../data", 'CACHE', 'ST3DNet', 'Rome16x8_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))
f = open(filename, 'rb')
X_train = pickle.load(f)
Y_train = pickle.load(f)
X_test = pickle.load(f)
Y_test = pickle.load(f)
mmn = pickle.load(f)
external_dim = pickle.load(f)
timestamp_train = pickle.load(f)
timestamp_test = pickle.load(f)

for i in X_train:
    print(i.shape)

Y_train = mmn.inverse_transform(Y_train)  # X is MaxMinNormalized, Y is real value
Y_test = mmn.inverse_transform(Y_test)

c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
t_conf = (len_trend, nb_flow, map_height,
          map_width) if len_trend > 0 else None

# build model
model = ST3DNet(c_conf=c_conf, t_conf=t_conf, external_dim=external_dim, nb_residual_unit=nb_residual_unit)

# load weights
model_fname = 'TaxiNYC2.c6.p0.t4.resunits_5.lr_0.00095.batchsize_16.best.h5'
model.load_weights(os.path.join('../best_models', 'ST3DNet', model_fname))

# predict
Y_pred = model.predict(X_test)  # compute predictions

# evaluate
score = evaluate(Y_test, Y_pred)  # evaluate performance

# save to csv
csv_name = os.path.join('results', f'TL_taxiNY_roma16x8_results.csv')
if not os.path.isfile(csv_name):
    if os.path.isdir('results') is False:
        os.mkdir('results')
    with open(csv_name, 'a', encoding = "utf-8") as file:
        file.write(
                'rsme_in,rsme_out,rsme_tot,'
                'mape_in,mape_out,mape_tot,'
                'ape_in,ape_out,ape_tot'
                )
        file.write("\n")
        file.close()
with open(csv_name, 'a', encoding = "utf-8") as file:
    file.write(f'{score[0]},{score[1]},{score[2]},{score[3]},'
            f'{score[4]},{score[5]},{score[6]},{score[7]},{score[8]}'
            )
    file.write("\n")
    file.close()

# save real vs predicted
fname = 'st3dnet_RomaNord16x8.h5'
h5 = h5py.File(fname, 'w')
h5.create_dataset('Y_real', data=Y_test)
h5.create_dataset('Y_pred', data=Y_pred)
h5.create_dataset('timestamps', data=timestamp_test)
h5.create_dataset('max', data=mmn._max)
h5.close()
