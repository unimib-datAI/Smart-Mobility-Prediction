from __future__ import print_function
import os
import _pickle as pickle
import numpy as np
import math
import h5py
from sklearn.model_selection import ParameterGrid

from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from star.model import *
import star.metrics as metrics
from star import TaxiNYC
from star.evaluation import evaluate

# parameters
DATAPATH = '../data' 
nb_epoch = 100  # number of epoch at training stage
# nb_epoch_cont = 150  # number of epoch at training (cont) stage
batch_size = [16, 32, 64]  # batch size
T = 24  # number of time intervals in one day
CACHEDATA = True  # cache data or NOT

lr = [0.00015, 0.00035]  # learning rate
# len_closeness = 3  # length of closeness dependent sequence
# len_period = 1  # length of peroid dependent sequence
# len_trend = 1  # length of trend dependent sequence
len_cpt = [[3,1,1]]
nb_residual_unit = [2,4,6]   # number of residual units

nb_flow = 2  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test, 
days_test = 7*4
len_test = T*days_test
len_val = 2*len_test

map_height, map_width = 16, 8  # grid size

path_cache = os.path.join(DATAPATH, 'CACHE', 'STAR')  # cache path
path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

def build_model(len_c, len_p, len_t, nb_flow, map_height, map_width,
                external_dim, nb_residual_unit, bn, bn2=False, save_model_pic=False, lr=0.00015):
    c_conf = (len_c, nb_flow, map_height,
              map_width) if len_c > 0 else None
    p_conf = (len_p, nb_flow, map_height,
              map_width) if len_p > 0 else None
    t_conf = (len_t, nb_flow, map_height,
              map_width) if len_t > 0 else None

    model = STAR(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                     external_dim=external_dim, nb_residual_unit=nb_residual_unit, bn=bn, bn2=bn2)
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
    X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test = [], [], [], [], [], [], [], []
    for i in range(num):
        X_train_all.append(f['X_train_all_%i' % i].value)
        X_train.append(f['X_train_%i' % i].value)
        X_val.append(f['X_val_%i' % i].value)
        X_test.append(f['X_test_%i' % i].value)
    Y_train_all = f['Y_train_all'].value
    Y_train = f['Y_train'].value
    Y_val = f['Y_val'].value
    Y_test = f['Y_test'].value
    external_dim = f['external_dim'].value
    timestamp_train_all = f['T_train_all'].value
    timestamp_train = f['T_train'].value
    timestamp_val = f['T_val'].value
    timestamp_test = f['T_test'].value
    f.close()

    return X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test, mmn, external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test

def cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test, external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X_train_all))

    for i, data in enumerate(X_train_all):
        h5.create_dataset('X_train_all_%i' % i, data=data)
    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    for i, data in enumerate(X_val):
        h5.create_dataset('X_val_%i' % i, data=data)
    # for i, data in enumerate(Y_train):
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)

    h5.create_dataset('Y_train_all', data=Y_train_all)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_val', data=Y_val)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train_all', data=timestamp_train_all)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_val', data=timestamp_val)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.close()

# build grid for grid search
params = {
    'len_cpt': len_cpt,
    'batch_size': batch_size,
    'lr': lr,
    'nb_residual_unit': nb_residual_unit
}

grid = ParameterGrid(params)
# print(grid)


for i in range(len(grid)):
    # extract current grid params
    len_c, len_p, len_t = grid[i]['len_cpt']
    nb_res_unit = grid[i]['nb_residual_unit']
    lr = grid[i]['lr']
    batch_size = grid[i]['batch_size']

    print('Step [{}/{}], len_c {}, len_p {}, len_t {}, res_unit {}, lr {}, batch_size {}'
        .format(i+1, len(grid), len_c, len_p, len_t, nb_res_unit, lr, batch_size))

    # load data
    print("loading data...")
    fname = os.path.join(path_cache, 'TaxiNYC_C{}_P{}_T{}.h5'.format(
        len_c, len_p, len_t))
    if os.path.exists(fname) and CACHEDATA:
        X_train_all, Y_train_all, X_train, Y_train, \
        X_val, Y_val, X_test, Y_test, mmn, external_dim, \
        timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = read_cache(
            fname)
        print("load %s successfully" % fname)
    else:
        X_train_all, Y_train_all, X_train, Y_train, \
        X_val, Y_val, X_test, Y_test, mmn, external_dim, \
        timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = TaxiNYC.load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_c, len_period=len_p, len_trend=len_t, len_test=len_test,
            len_val=len_val, preprocess_name='preprocessing_taxinyc.pkl', meta_data=True, datapath=DATAPATH)
        if CACHEDATA:
            cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
                  external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test)

    # print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
    print('=' * 10)

    iterations = 1
    for iteration in range(0, iterations):
        # build model
        print(f'Iteration {iteration}')
        tf.keras.backend.set_image_data_format('channels_first')

        model = build_model(len_c, len_p, len_t, nb_flow, map_height,
                            map_width, external_dim, nb_res_unit,
                            bn=True,
                            bn2=True,
                            save_model_pic=False,
                            lr=lr
                            )

        hyperparams_name = 'TaxiNYC.c{}.p{}.t{}.resunit_{}.lr_{}.batchsize_{}.iter{}'.format(
            len_c, len_p, len_t, nb_res_unit, lr, batch_size, iteration)
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
        csv_name = os.path.join('results','star_bikeNYC_results.csv')
        if not os.path.isfile(csv_name):
            if os.path.isdir('results') is False:
                os.mkdir('results')
            with open(csv_name, 'a', encoding = "utf-8") as file:
                file.write(
                  'len_closeness,len_period,len_trend,'
                  'nb_residual_unit,'
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
              f'{nb_res_unit},'
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