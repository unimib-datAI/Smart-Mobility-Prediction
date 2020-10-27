from src import TaxiBJ
import numpy as np
import time
import os

from utils import cache, read_cache, build_model 

np.random.seed(1337)  # for reproducibility

# PARAMETERS
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

path_cache = os.path.join(DATAPATH, 'CACHE', 'Autoencoder')  # cache path
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
        fname)
    print("load %s successfully" % fname)
else:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = TaxiBJ.load_data(
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
model = build_model(len_closeness, len_period, len_trend, save_model_pic=False)