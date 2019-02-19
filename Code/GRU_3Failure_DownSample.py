
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
from keras.layers import LSTM, GRU
import keras.callbacks
from keras import backend as K


from LoadAndProcessCSVFile import TimeStepSize
from LoadAndProcessCSVFile import loadAndProcessRawData

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
df = pd.DataFrame();
y = pd.DataFrame();

for i in range(1,2):
    
    sensorFilePath = '../Data/train/0{}_M01_DC_train.csv'.format(i)
    faultsFilePath = '../Data/train/train_faults/0{}_M01_train_fault_data.csv'.format(i)
    ttfFilePath = '../Data/train/train_ttf/0{}_M01_DC_train.csv'.format(i) 
    df_tmp, y_tmp = loadAndProcessRawData(sensorFilePath, faultsFilePath, ttfFilePath)

    df = df.append(df_tmp)
    y = [y,y_tmp]
    y = pd.concat(y)
    
    sensorFilePath = '../Data/train/0{}_M02_DC_train.csv'.format(i,i)
    faultsFilePath = '../Data/train/train_faults/0{}_M02_train_fault_data.csv'.format(i,i)
    ttfFilePath = '../Data/train/train_ttf/0{}_M02_DC_train.csv'.format(i,i) 
    df_tmp, y_tmp = loadAndProcessRawData(sensorFilePath, faultsFilePath, ttfFilePath)

    df = df.append(df_tmp)
    y = [y,y_tmp]
    y = pd.concat(y)



#------------------------------------------------------------------------------
# scale data for better performance
df_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
y_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
feature = df_scaler.fit_transform(df)
label = y_scaler.fit_transform(y)


#------------------------------------------------------------------------------
# split data for train, validate, and test
x, X_test, y, y_test = train_test_split(feature,label,test_size=0.2,train_size=0.8)
X_train, X_valid, y_train, y_valid = train_test_split(x,y,test_size = 0.1,train_size =0.9)

#------------------------------------------------------------------------------
# LSTM
X_train = X_train.reshape((X_train.shape[0], TimeStepSize, 22))
X_valid = X_valid.reshape((X_valid.shape[0], TimeStepSize, 22))
X_all = feature.reshape((feature.shape[0], TimeStepSize, 22))

#------------------------------------------------------------------------------
# Train

model = Sequential()
model.add(GRU(10, return_sequences=True,  input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(GRU(10, return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(10))
model.add(Dense(1))
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss=customLoss, optimizer='adam')
# Early stopping
es = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
history = model.fit(X_train, y_train, epochs=100, batch_size=256, \
                    validation_data=(X_valid, y_valid), verbose=2, shuffle=False)

#------------------------------------------------------------------------------
# Visualize
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

yhat = model.predict(X_all)
y_pred = y_scaler.inverse_transform(yhat)
y_real = y_scaler.inverse_transform(label)
plt.figure()
#t=np.arange(len(yhat))/len(label)*max(ttf_fault1['time'])/3600
#scale = 1/len(label)*max(ttf_fault1['TTF_FlowCool Pressure Dropped Below Limit'])/3600;
plt.plot(y_real[:,0],label="Real Data")
plt.plot(y_pred[:,0],label="Predicted")
plt.xlabel("Time (hour)")
plt.ylabel("Remaining Life (hour)")
plt.title("Predicted Remaining Life v.s. Real Remaining Life")
plt.legend();
plt.show()
