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
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, GRU
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

## Train

model = Sequential()
model.add(GRU(10, return_sequences=True,  input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(GRU(10, return_sequences=True))
model.add(GRU(10))
model.add(Dense(10))
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss=customLoss, optimizer='adam')
# Early stopping
es = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
history = model.fit(X_train, y_train, epochs=500, batch_size=256, \
                    validation_data=(X_valid, y_valid), verbose=2, shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# scale back the outputs
yhat = model.predict(X_train)
y_pred = y_scaler.inverse_transform(yhat)
y_real = y_scaler.inverse_transform(y_train)
plt.figure()
t=np.arange(len(yhat))/len(label)*max(ttf_fault1['time'])/3600
scale = 1/len(label)*max(ttf_fault1['TTF_FlowCool Pressure Dropped Below Limit'])/3600;
plt.plot(t,y_real[:,0]*scale,label="Real Data")
plt.plot(t,y_pred[:,0]*scale,label="Predicted")
plt.xlabel("Time (hour)")
plt.ylabel("Remaining Life (hour)")
plt.title("Predicted Remaining Life v.s. Real Remaining Life")
plt.legend();
plt.show()
# =============================================================================
# #------------------------------------------------------------------------------
# # Check correlation between features and labels
# def spearman(frame, features):
#     spr = pd.DataFrame()
#     spr['feature'] = features
#     spr['spearman'] = [frame[f].corr(frame['TTF_FlowCool Pressure Dropped Below Limit'], 'spearman') for f in features]
#     spr = spr.sort_values('spearman')
#     plt.figure(figsize=(6, 0.25*len(features)))
#     sns.barplot(data=spr, y='feature', x='spearman', orient='h')
# features = df.columns[0:18]
# spearman(df, features)
# =============================================================================
yhat = model.predict(X_all)
y_pred = y_scaler.inverse_transform(yhat)
y_real = y_scaler.inverse_transform(label)
plt.figure()
t=np.arange(len(yhat))/len(label)*max(ttf_fault1['time'])/3600
scale = 1/len(label)*max(ttf_fault1['TTF_FlowCool Pressure Dropped Below Limit'])/3600;
plt.plot(t,y_real[:,0]*scale,label="Real Data")
plt.plot(t,y_pred[:,0]*scale,label="Predicted")
plt.xlabel("Time (hour)")
plt.ylabel("Remaining Life (hour)")
plt.title("Predicted Remaining Life v.s. Real Remaining Life")
plt.legend();
plt.show()
