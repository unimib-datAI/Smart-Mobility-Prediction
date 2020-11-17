from __future__ import print_function
import os
import _pickle as pickle
import numpy as np
import math
import h5py
from sklearn.model_selection import ParameterGrid

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from src.model import build_model
import src.metrics as metrics
from src.datasets import TaxiNYC
from src.evaluation import evaluate
from cache_utils import cache, read_cache

# parameters
DATAPATH = '../data' 
nb_epoch = 100  # number of epoch at training stage
# nb_epoch_cont = 150  # number of epoch at training (cont) stage
batch_size = [16, 64]  # batch size
T = 24  # number of time intervals in one day
CACHEDATA = True  # cache data or NOT

lr = [0.0015, 0.00015]  # learning rate
# len_closeness = 3  # length of closeness dependent sequence
# len_period = 1  # length of peroid dependent sequence
# len_trend = 1  # length of trend dependent sequence
len_cpt = [[2,0,1]]
lstm = [350, 500]
lstm_number = [2, 3]

nb_flow = 2  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test, 
days_test = 7*4
len_test = T*days_test
len_val = 2*len_test

map_height, map_width = 16, 8  # grid size

path_cache = os.path.join(DATAPATH, 'CACHE', '3D-CLoST')  # cache path
path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

# build grid for grid search
params = {
    'len_cpt': len_cpt,
    'batch_size': batch_size,
    'lr': lr,
    'lstm': lstm,
    'lstm_number': lstm_number
}

grid = ParameterGrid(params)

for i in range(len(grid)):
    # extract current grid params
    len_c, len_p, len_t = grid[i]['len_cpt']
    lstm = grid[i]['lstm']
    lstm_number = grid[i]['lstm_number']
    lr = grid[i]['lr']
    batch_size = grid[i]['batch_size']

    print('Step [{}/{}], len_c {}, len_p {}, len_t {}, lstm_unit {}, lstm_number {}, lr {}, batch_size {}'
        .format(i+1, len(grid), len_c, len_p, len_t, lstm, lstm_number, lr, batch_size))

    # load data
    print("loading data...")
    fname = os.path.join(path_cache, 'TaxiNYC_C{}_P{}_T{}.h5'.format(
        len_c, len_p, len_t))
    if os.path.exists(fname) and CACHEDATA:
        X_train_all, Y_train_all, X_train, Y_train, \
        X_val, Y_val, X_test, Y_test, mmn, external_dim, \
        timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask = read_cache(
            fname, 'preprocessing_taxinyc.pkl')
        print("load %s successfully" % fname)
    else:
        X_train_all, Y_train_all, X_train, Y_train, \
        X_val, Y_val, X_test, Y_test, mmn, external_dim, \
        timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask = TaxiNYC.load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_c, len_period=len_p, len_trend=len_t, len_test=len_test,
            len_val=len_val, preprocess_name='preprocessing_taxinyc.pkl', meta_data=True, datapath=DATAPATH)
        if CACHEDATA:
            cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                  external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask)

    # print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
    print('=' * 10)

    iterations = 1
    for iteration in range(0, iterations):
        # build model
        print(f'Iteration {iteration}')

        model = build_model('NY', X_train,  Y_train, conv_filt=64, kernel_sz=(2,3,3), 
                    mask=mask, lstm=lstm, lstm_number=lstm_number, add_external_info=True,
                    lr=lr, save_model_pic=None)

        hyperparams_name = 'TaxiNYC.c{}.p{}.t{}.lstm_{}.lstmnumber_{}.lr_{}.batchsize_{}.iter{}'.format(
            len_c, len_p, len_t, lstm, lstm_number, lr, batch_size, iteration)
        fname_param = os.path.join(path_model, '{}.best.h5'.format(hyperparams_name))
        print(hyperparams_name)

        early_stopping = EarlyStopping(monitor='val_rmse', patience=25, mode='min')
        model_checkpoint = ModelCheckpoint(
            fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

        print('=' * 10)
        # train model
        np.random.seed(i*18)
        tf.random.set_seed(i*18)
        print("training model...")
        history = model.fit(X_train, Y_train,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            validation_data=(X_val,Y_val),
                            callbacks=[early_stopping, model_checkpoint],
                            verbose=0)
        model.save_weights(os.path.join(
            path_model, '{}.h5'.format(hyperparams_name)), overwrite=True)
        pickle.dump((history.history), open(os.path.join(
            path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))

        print('=' * 10)

        # evaluate model
        print('evaluating using the model that has the best loss on the valid set')
        model.load_weights(fname_param) # load best weights for current iteration
        
        Y_pred = model.predict(X_test) # compute predictions

        score = evaluate(Y_test, Y_pred, mmn, rmse_factor=1) # evaluate performance

        # save to csv
        csv_name = os.path.join('results','3DCLoST_bikeNYC_results.csv')
        if not os.path.isfile(csv_name):
            if os.path.isdir('results') is False:
                os.mkdir('results')
            with open(csv_name, 'a', encoding = "utf-8") as file:
                file.write(
                  'len_closeness,len_period,len_trend,'
                  'lstm,'
                  'lstm_number,'
                  'learning rate,'
                  'batch_size,'
                  'iteration,'
                  'rsme_in,rsme_out,rsme_tot,'
                  'mape_in,mape_out,mape_tot,'
                  'ape_in,ape_out,ape_tot'
                )
                file.write("\n")
                file.close()
        with open(csv_name, 'a', encoding = "utf-8") as file:
            file.write(
              f'{len_c},{len_p},{len_t},'
              f'{lstm},'
              f'{lstm_number},'
              f'{lr},'
              f'{batch_size},'
              f'{iteration},'
              f'{score[0]},{score[1]},{score[2]},'
              f'{score[3]},{score[4]},{score[5]},'
              f'{score[6]},{score[7]},{score[8]}'
            )
            file.write("\n")
            file.close()
        K.clear_session()