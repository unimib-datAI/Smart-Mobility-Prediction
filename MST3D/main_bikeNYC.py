from __future__ import print_function
import os
import sys
import pickle
import time
import numpy as np
import h5py
import math

import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    Input,
    Conv3D,
    MaxPool3D,
    Dropout,
    Flatten,
    Activation,
    Add,
    Dense,
    Reshape,
    BatchNormalization
)
from keras.models import Model

import deepst.metrics as metrics
from deepst.datasets import BikeNYC


np.random.seed(1337)  # for reproducibility

# parameters
DATAPATH = '../data'
CACHEDATA = True  # cache data or NOT
path_cache = os.path.join(DATAPATH, 'CACHE', 'MST3D')  # cache path
nb_epoch = 200  # number of epoch at training stage
nb_epoch_cont = 100  # number of epoch at training (cont) stage
batch_size = 64  # batch size
T = 24  # number of time intervals in one day
lr = 0.0002  # learning rate
len_closeness = 4  # length of closeness dependent sequence - should be 6
len_period = 4  # length of peroid dependent sequence
len_trend = 4  # length of trend dependent sequence

nb_flow = 2  # there are two types of flows: inflow and outflow

# divide data into two subsets: Train & Test, of which the test set is the
# last 10 days
days_test = 10
len_test = T * days_test
map_height, map_width = 16, 8  # grid size
# For NYC Bike data, there are 81 available grid-based areas, each of
# which includes at least ONE bike station. Therefore, we modify the final
# RMSE by multiplying the following factor (i.e., factor).
nb_area = 81
m_factor = math.sqrt(1. * map_height * map_width / nb_area)

path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)


# custom layer for branches fusion
class LinearLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(LinearLayer, self).__init__()
    # self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel1 = self.add_weight("kernel1", input_shape[0][1:])
    self.kernel2 = self.add_weight("kernel2", input_shape[1][1:])
    self.kernel3 = self.add_weight("kernel3", input_shape[2][1:])


  def call(self, inputs):
    return (
        tf.math.multiply(inputs[0], self.kernel1)
        + tf.math.multiply(inputs[1], self.kernel2)
        + tf.math.multiply(inputs[2], self.kernel3)

'''
    MST3D implementation for BikeNYC
'''
def mst3d(len_c, len_p, len_t, nb_flow=2, map_height=16, map_width=8, external_dim=8):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    external_dim
    '''

    # main input
    main_inputs = []
    outputs = []
    for len in [len_c, len_p, len_t]:
        if len is not None:
            input = Input(shape=(len, map_height, map_width, nb_flow))
            main_inputs.append(input)
            
            # the first convolutional layer has 32 filters and kernel size of (2,3,3)
            # set stride to (2,1,1) to reduce depth
            stride = (1,1,1)
            nb_filters = 32
            kernel_size = (2,3,3)

            conv1 = Conv3D(nb_filters, kernel_size, padding='same', activation='relu', strides=stride)(input)
            maxPool1 = MaxPool3D((1,2,2))(conv1)
            maxPool1 = BatchNormalization()(maxPool1)
            dropout1 = Dropout(0.25)(maxPool1)
            print(dropout1.shape)

            # the second layers have 64 filters
            nb_filters = 64
            
            conv2 = Conv3D(nb_filters, kernel_size, padding='same', activation='relu', strides=stride)(dropout1)
            maxPool2 = MaxPool3D((1,2,2))(conv2)
            maxPool2 = BatchNormalization()(maxPool2)
            dropout2 = Dropout(0.25)(maxPool2)
            print(dropout2.shape)

            outputs.append(dropout2)

    # parameter-matrix-based fusion
    fusion = LinearLayer()(outputs)
    flatten = Flatten()(fusion)

    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(10)(external_input)
        embedding = Activation('relu')(embedding)
        # h1 = Dense(nb_filters * 2 * map_height/4 * map_width/4)(embedding)
        h1 = Dense(flatten.shape[1])(embedding)
        activation = Activation('relu')(h1)
        main_output = Add()([flatten, activation])

    # reshape and tanh activation
    main_output = Dense(nb_flow * map_height * map_width)(main_output)
    main_output = Reshape((map_height, map_width, nb_flow))(main_output)
    main_output = Activation('tanh')(main_output)

    model = Model(main_inputs, main_output)

    return model

def build_model(save_model_pic=False):
    model = mst3d(len_closeness, len_period, len_trend, nb_flow, map_height, map_width, external_dim)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    # model.summary()
    if (save_model_pic):
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='BikeNYC_model.png', show_shapes=True)
    return model

def read_cache(fname):
    mmn = pickle.load(open('preprocessing.pkl', 'rb'))

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

# main code
# load data
print("loading data...")
ts = time.time()
fname = os.path.join(path_cache, 'BikeNYC_C{}_P{}_T{}.h5'.format(
    len_closeness, len_period, len_trend))
if os.path.exists(fname) and CACHEDATA:
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(
        fname)
    print("load %s successfully" % fname)
else:
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        preprocess_name='preprocessing.pkl', meta_data=True, datapath=DATAPATH)
    if CACHEDATA:
        cache(fname, X_train, Y_train, X_test, Y_test,
              external_dim, timestamp_train, timestamp_test)

print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

print('=' * 10)

# compile model
print("compiling model...")
print(
    "**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")
ts = time.time()
model = build_model(save_model_pic=True)
hyperparams_name = 'BikeNYC.c{}.p{}.t{}.lr{}'.format(
    len_closeness, len_period, len_trend, lr)
fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

early_stopping = EarlyStopping(monitor='val_rmse', patience=2, mode='min')
model_checkpoint = ModelCheckpoint(
    fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

print("\nelapsed time (compiling model): %.3f seconds\n" %
      (time.time() - ts))

print('=' * 10)

# train model
print("training model...")
ts = time.time()
history = model.fit(X_train, Y_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=1)
model.save_weights(os.path.join(
    'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
pickle.dump((history.history), open(os.path.join(
    path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))
print('=' * 10)

# evaluate
print('evaluating using the model that has the best loss on the valid set')

model.load_weights(fname_param)
score = model.evaluate(
    X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
        (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))