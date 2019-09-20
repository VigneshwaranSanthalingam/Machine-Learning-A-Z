 -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 20:02:05 2018

@author: vikuv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

timestamps = [10,20,30,40,50,60]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
mcp = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard(log_dir = 'rnn.log')

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']),axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs_scaled = sc.transform(inputs)

for t in timestamps:
  X_train = []
  y_train = []
  for i in range(t, len(dataset_train)):
      X_train.append(training_set_scaled[i-t:i, 0])
      y_train.append(training_set_scaled[i, 0])
  X_train, y_train = np.array(X_train), np.array(y_train)
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
  
  inputs = dataset_total[len(dataset_total)-len(dataset_test)-t:].values
  inputs = inputs.reshape(-1,1)
  inputs_scaled = sc.transform(inputs)
  
  X_test = []
  for j in range(t, len(inputs)):
     X_test.append(inputs_scaled[j-t:j, 0])
  X_test = np.array(X_train)
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
  print("timestamps : %d" % (t))
  regressor = Sequential()
  regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
  regressor.add(Dropout(0.2))
  regressor.add(LSTM(units = 50, return_sequences = True))
  regressor.add(Dropout(0.2))
  regressor.add(LSTM(units = 50, return_sequences = True))
  regressor.add(Dropout(0.2))
  regressor.add(LSTM(units = 50))
  regressor.add(Dropout(0.2))
  regressor.add(Dense(units = 1))
  regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
  history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 32, verbose = 1, validation_split = 0.2, callbacks = [es, rlr, mcp, tb])
  
  predicted_stock = regressor.predict(X_test)
  predicted_stock = sc.inverse_transform(predicted_stock)
  plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
  plt.plot(predicted_stock, color = 'blue', label = 'Predicted Google Stock Price')
  plt.title('Google Stock Price Prediction')
  plt.xlabel('Time')
  plt.ylabel('Google Stock Price')
  plt.legend()
  plt.show()

