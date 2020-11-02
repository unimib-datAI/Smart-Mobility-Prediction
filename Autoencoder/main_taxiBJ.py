import numpy as np
import time
import os
import pickle as pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint

from src import TaxiBJ, TaxiBJ3d
from utils import cache, read_cache, build_model 

np.random.seed(1337)  # for reproducibility

# PARAMETERS
model_name = 'model3'
DATAPATH = '../data'  
CACHEDATA = True  # cache data or NOT
nb_epoch = 100 # number of epoch at training stage
nb_epoch_cont =  100 # number of epoch at training (cont) stage
batch_size = 16  # batch size
T = 48  # number of time intervals in one day
lr = 0.00015 # learning rate

len_closeness = 3 # length of closeness dependent sequence
len_period = 1 # length of peroid dependent sequence
len_trend = 1 # length of trend dependent sequence
nb_flow = 2  # there are two types of flows: inflow and outflow
# divide data into two subsets: Train & Test, of which the test set is the
# last 4 weeks
days_test = 7*4
len_test = T*days_test
len_val = 2*len_test
map_height, map_width = 32, 32  # grid size

path_log = 'log_BJ'
muilt_step = False

cache_folder = 'Autoencoder/model3' if model_name == 'model3' else 'Autoencoder'
path_cache = os.path.join(DATAPATH, 'CACHE', cache_folder)  # cache path
path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

# load data
if muilt_step:
    dic_rmse={}
    list_muilt_rmse=[]
print("loading data...")
ts = time.time()
fname = os.path.join(path_cache, 'TaxiBJ_C{}_P{}_T{}.h5'.format(
    len_closeness, len_period, len_trend))
if os.path.exists(fname) and CACHEDATA:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = read_cache(
        fname, 'preprocessing_bj.pkl')
    print("load %s successfully" % fname)
else:
    if (model_name == 'model3'):
        load_data = TaxiBJ3d.load_data
    else:
        load_data = TaxiBJ.load_data
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        len_val=len_val, preprocess_name='preprocessing_bj.pkl', meta_data=True, meteorol_data=True, holiday_data=True, datapath=DATAPATH)
    if CACHEDATA:
        cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
              external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test)
i = 0
print(external_dim)
print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

# build model
model = build_model(len_closeness, len_period, len_trend, model=model_name,
                    external_dim=external_dim, lr=lr,
                    # save_model_pic=f'TaxiBJ_{model_name}'
                    )
hyperparams_name = '{}.TaxiBJ.c{}.p{}.t{}.lr{}'.format(
    model_name, len_closeness, len_period, len_trend, lr)
fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

early_stopping = EarlyStopping(monitor='val_rmse', patience=2, mode='min')
model_checkpoint = ModelCheckpoint(
    fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

# train model
print("training model...")
ts = time.time()
history = model.fit(X_train, Y_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    validation_data=(X_val,Y_val),
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=2)
model.save_weights(os.path.join(
    'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
pickle.dump((history.history), open(os.path.join(
    path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

# evaluate
model.load_weights(fname_param)
score = model.evaluate(
    X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
        (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))