import numpy as np
from numpy import inf
import keras
import matplotlib
import math
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
#from tensorflow_probability.python.layers.dense_variational import DenseFlipout

import h5py
from tensorflow_probability import distributions as tfd

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from IPython.display import FileLink, FileLinks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

from keras.utils import to_categorical, plot_model
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard #for keras+theano
#from tensorflow.keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard #for tensorflow, e.g tfp #DOES NOT WORK
from keras.optimizers import Adam
from keras import metrics, regularizers
import pickle
from copy import deepcopy
import os
from functions import *
from PredictExternal import *
import time

import pandas as pd
from clearml import Task

def TrainNetwork(parameters, inputfolder, outputfolder):
    #clearml
    #task = Task.init(project_name="Ksenia", task_name=dict_to_str(parameters))
    # Get parameters
    layers=parameters['layers']
    batch_size=parameters['batchsize']
    regmethod=parameters['regmethod']
    regrate=parameters['regrate']
    batchnorm=parameters['batchnorm']
    epochs=parameters['epochs']
    learningrate = parameters['learningrate']
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight=parameters['eqweight']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    train_new_model = True
    try:
        #model = keras.models.load_model('output/'+tag+'/model.h5')
        model = keras.models.load_model(outputfolder+'/model.h5')
        train_new_model = False
    except:
        pass
    if train_new_model: print 'Couldn\'t find the model "%s", a new one will be trained!' % (tag)
    else:
        print 'Found the model "%s", not training a new one, go on to next function.' % (tag)
        return
    if not os.path.isdir(outputfolder): os.makedirs(outputfolder)

    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, signal_eventweights, signal_normweights = load_data(parameters, inputfolder=inputfolder, filepostfix='')

    # Define the network
    model = Sequential()
    kernel_regularizer = None
    if regmethod == 'L1':
        kernel_regularizer=regularizers.l1(regrate)
    elif regmethod == 'L2':
        kernel_regularizer=regularizers.l2(regrate)


    print 'Number of input variables: %i' % (input_train.shape[1])
    model.add(Dense(layers[0], activation='relu', input_shape=(input_train.shape[1],), kernel_regularizer=kernel_regularizer))
    if regmethod == 'dropout': model.add(Dropout(regrate))
    if batchnorm: model.add(BatchNormalization())

    for i in layers[1:len(layers)+1]:
        model.add(Dense(i, activation='relu', kernel_regularizer=kernel_regularizer))
        if batchnorm: model.add(BatchNormalization())
        if regmethod == 'dropout': model.add(Dropout(regrate))


    model.add(Dense(labels_train.shape[1], activation='softmax', kernel_regularizer=kernel_regularizer))
    #model.add(Dense(labels_train.shape[1], activation='sigmoid', kernel_regularizer=kernel_regularizer))
    print 'Number of output classes: %i' % (labels_train.shape[1])


    # Train the network
    opt = keras.optimizers.Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=1e-6, decay=0.0, amsgrad=False)
    mymetrics = [metrics.categorical_accuracy]
    # mymetrics = [metrics.categorical_accuracy, metrics.mean_squared_error, metrics.categorical_crossentropy, metrics.kullback_leibler_divergence, metrics.cosine_proximity]
    model.compile(loss='categorical_crossentropy', optimizer=opt, weighted_metrics=mymetrics)
    #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=mymetrics)
    print model.summary()

    period = epochs / 5
    checkpointer = ModelCheckpoint(filepath=outputfolder+'/model_epoch{epoch:02d}.h5', verbose=1, save_best_only=False, period=period)
    checkpointer_everymodel = ModelCheckpoint(filepath=outputfolder+'/model_epoch{epoch:02d}.h5', verbose=1, save_best_only=False, mode='auto', period=1)
    checkpoint_bestmodel = ModelCheckpoint(filepath=outputfolder+'/model_best.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=100, verbose=0, mode='min', baseline=None, restore_best_weights=True)
    LRreducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_delta=0.001, mode='min')
    weights_train, weights_test = sample_weights_train, sample_weights_test
    if not eqweight:
        weights_train, weights_test = eventweights_train, eventweights_test
    print 'labels_train= %i' % (labels_train.shape[1])
    print labels_train

    #### validation data = test set (obwohl Konventioniell hier validation data vewendet wird) -> 'Umdefinieren' test<->validation
    model.fit(input_train, labels_train, sample_weight=weights_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(input_test, labels_test, weights_test), callbacks=[checkpointer_everymodel, checkpoint_bestmodel], verbose=2)


    model.save(outputfolder+'/model.h5')
    with open(outputfolder+'/model_history.pkl', 'w') as f:
        pickle.dump(model.history.history, f)


    #PredictExternal(parameters, inputfolder='input/'+classtag, outputfolder=outputfolder, filepostfix='')


def _prior_normal_fn(sigma, dtype, shape, name, trainable, add_variable_fn):
    """Normal prior with mu=0 and sigma=sigma. Can be passed as an argument to
    the tpf.layers
    """
    del name, trainable, add_variable_fn

    dist = tfd.Normal(loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(sigma))
    batch_ndims = tf.size(input=dist.batch_shape_tensor())
    return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
