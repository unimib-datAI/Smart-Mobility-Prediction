from __future__ import print_function
import os
import sys
import pickle
import time
import numpy as np
import h5py
import math
from sklearn.model_selection import ParameterGrid

import tensorflow as tf
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

import deepst.metrics as metrics
from deepst.datasets import TaxiNYC
from deepst.model import mst3d_nyc
from deepst.evaluation import evaluate

# parameters
DATAPATH = '../data'
CACHEDATA = True  # cache data or NOT
path_cache = os.path.join(DATAPATH, 'CACHE', 'MST3D')  # cache path
nb_epoch = 100  # number of epoch at training stage
# nb_epoch_cont = 100  # number of epoch at training (cont) stage
batch_size = [16, 32, 64]  # batch size
T = 24  # number of time intervals in one day
lr = [0.00015, 0.00035]  # learning rate
# len_closeness = 4  # length of closeness dependent sequence - should be 6
# len_period = 4  # length of peroid dependent sequence
# len_trend = 4  # length of trend dependent sequence
len_cpt = [[4,4,4]]
nb_flow = 2  # there are two types of flows: inflow and outflow

# divide data into two subsets: Train & Test, of which the test set is the
# last 10 days
days_test = 10
len_test = T * days_test
map_height, map_width = 16, 8  # grid size

path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

def build_model(len_c, len_p, len_t, nb_flow, map_height, map_width,
                external_dim, save_model_pic=False, lr=0.00015):
    model = mst3d_nyc(
      len_c, len_p, len_t,
      nb_flow, map_height, map_width,
      external_dim
    )
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    # model.summary()
    if (save_model_pic):
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='TaxiNYC_model.png', show_shapes=True)

    return model

def read_cache(fname):
    mmn = pickle.load(open('preprocessing_taxinyc.pkl', 'rb'))

    f = h5py.File(fname, 'r')
    num = int(f['num'].value)
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(num):
        X_train.append(f['X_train_%i' % i].value)
        X_test.append(f['X_test_%i' % i].value)
    Y_train = f['Y_train'].value
    Y_test = f['Y_test'].value
    external_dim = f['external_dim'].value
    timestamp_train = f['T_train'].value
    timestamp_test = f['T_test'].value
    f.close()

    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test

def cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X_train))

    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    # for i, data in enumerate(Y_train):
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.close()

# build grid for grid search
params = {
    'len_cpt': len_cpt,
    'batch_size': batch_size,
    'lr': lr,
}

grid = ParameterGrid(params)
# print(grid)

for i in range(len(grid)):
    # extract current grid params
    len_c, len_p, len_t = grid[i]['len_cpt']
    lr = grid[i]['lr']
    batch_size = grid[i]['batch_size']

    print('Step [{}/{}], len_c {}, len_p {}, len_t {}, lr {}, batch_size {}'
        .format(i+1, len(grid), len_c, len_p, len_t, lr, batch_size))

    # load data
    print("loading data...")
    fname = os.path.join(path_cache, 'TaxiNYC_C{}_P{}_T{}.h5'.format(
        len_c, len_p, len_t))
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(
            fname)
        print("load %s successfully" % fname)
    else:
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = TaxiNYC.load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_c, len_period=len_p, len_trend=len_t, len_test=len_test,
            preprocess_name='preprocessing_taxinyc.pkl', meta_data=True, datapath=DATAPATH)
        if CACHEDATA:
            cache(fname, X_train, Y_train, X_test, Y_test,
                  external_dim, timestamp_train, timestamp_test)

    print('=' * 10)

    # print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
    print('=' * 10)

    iterations = 1
    for iteration in range(0, iterations):
        # build model
        print(f'Iteration {iteration}')

        model = build_model(
            len_c, len_p, len_t, nb_flow, map_height,
            map_width, external_dim,
            save_model_pic=False,
            lr=lr
        )

        hyperparams_name = 'TaxiNYC.c{}.p{}.t{}.lr_{}.batchsize_{}.iter{}'.format(
            len_c, len_p, len_t, lr, batch_size, iteration)
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
                            validation_split=0.1,
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
        csv_name = os.path.join('results','mst3d_taxiNYC_results.csv')
        if not os.path.isfile(csv_name):
            if os.path.isdir('results') is False:
                os.mkdir('results')
            with open(csv_name, 'a', encoding = "utf-8") as file:
                file.write(
                  'len_closeness,len_period,len_trend,'
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