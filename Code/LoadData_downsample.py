# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 00:44:00 2019

@author: huach
"""

# -*- coding: utf-8 -*-

import random
random.seed(1234)
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import keras.callbacks
from keras import backend as K

def Error(y_pred, y_real):
    y_pred = np.nan_to_num(y_pred, copy = True)
    y_real = np.nan_to_num(y_real, copy = True)
    temp = np.exp(-0.001 * y_real) * np.abs(y_real - y_pred)
    error = np.sum(temp)
    return error

def customLoss(y_pred, y_real):
    return K.sum(K.exp(-0.001 * y_real) * K.abs(y_real - y_pred))
    
#------------------------------------------------------------------------------
# Read in Data

sensor_data = pd.read_csv("../Data/train/01_M01_DC_train.csv")
faults_data = pd.read_csv('../Data/train/train_faults/01_M01_train_fault_data.csv')
ttf_data = pd.read_csv('../Data/train/train_ttf/01_M01_DC_train.csv')

sensor_data = sensor_data.drop(['Tool'], axis = 1)
sensor_data = sensor_data.drop(['Lot'], axis = 1)

# =============================================================================
# sensor_data = sensor_data.loc[sensor_data.index %10 == 0]
# ttf_data = ttf_data.loc[ttf_data.index %10 == 0]
# =============================================================================
sensor_data.index = range(0,len(sensor_data))
ttf_data.index = range(0,len(ttf_data))

def cutoff(sensor_data, faults_data, ttf_data, column):
    # cut off the tail of the data set that with NaN ttf
    temp = faults_data[faults_data['fault_name'] == column]
    last_failure = temp['time'].values[-1]
    array = np.asarray(sensor_data['time'])
    closest_ind = (np.abs(array - last_failure)).argmin()
    if ((array[closest_ind] - last_failure) != np.abs(array[closest_ind] - last_failure)):
        ind = closest_ind + 1
    elif ((array[closest_ind] - last_failure) == 0):
        ind = closest_ind + 1
    else:
        ind = closest_ind
    sensor_data = sensor_data[:ind]
    ttf_data = ttf_data[:ind]
    faults_data = faults_data[faults_data['fault_name'] == column]
    return sensor_data, ttf_data, faults_data

sensor_fault1, ttf_fault1, faults_fault1 = cutoff(sensor_data, faults_data, \
                    ttf_data, 'FlowCool Pressure Dropped Below Limit')    

sensor_fault1 = sensor_fault1.fillna(method = 'ffill')
sensor_fault1['recipe'] = sensor_fault1['recipe'] + 200
label = ttf_fault1['TTF_FlowCool Pressure Dropped Below Limit']

#------------------------------------------------------------------------------
# down sample to data 
sampleRate = 60
    
df_select, y_select = sensor_fault1.ix[::sampleRate], label.ix[::sampleRate]
#------------------------------------------------------------------------------
# Shift dataset
def series_to_supervised(data, y, n_in=50, dropnan=True):
    data_col = []
    y_col = []
    for i in range (0, n_in):
        data_col.append(data.shift(i))
        y_col.append(y.shift(i))
    result = pd.concat(data_col, axis = 1)
    label = pd.concat(y_col, axis = 1)
    if dropnan:
        result = result[n_in:]
        label = label[n_in:]
    return result, label

df, y = series_to_supervised(df_select, y_select, 10, True)
df_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
y_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
feature = df_scaler.fit_transform(df)
label = y_scaler.fit_transform(y)

x, X_test, y, y_test = train_test_split(feature,label,test_size=0.2,train_size=0.8)
X_train, X_valid, y_train, y_valid = train_test_split(x,y,test_size = 0.25,train_size =0.75)

#------------------------------------------------------------------------------
# LSTM
X_train = X_train.reshape((X_train.shape[0], 10, 22))
X_valid = X_valid.reshape((X_valid.shape[0], 10, 22))
X_all = feature.reshape((feature.shape[0], 10, 22))