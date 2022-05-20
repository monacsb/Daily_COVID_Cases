# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:28:45 2022

@author: Mona
"""

import os
import pandas as pd 
import numpy as np 

#Visualization
import matplotlib.pyplot as plt

#Data preprocessing
from sklearn.preprocessing import MinMaxScaler 

#Model creation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import datetime

#Model Evaluation
from sklearn.metrics import mean_absolute_error

#Save scaler
import pickle

#%% Static 

TRAIN_PATH = os.path.join(os.getcwd(),'cases_malaysia_train.csv')
TEST_PATH = os.path.join(os.getcwd(),'cases_malaysia_test.csv')

MODEL_PATH = os.path.join(os.getcwd(),'cases_new.h5')
LOG_PATH = os.path.join(os.getcwd(),'log')

#%%EDA

#Step 1: Load Data

X_train = pd.read_csv(TRAIN_PATH)
X_test = pd.read_csv(TEST_PATH)

#Step 2: Data interpretation

#consist of 31 columns 
X_train.info()

X_train.describe()

X_train.describe().T

#Checking on null values
#Columns that contain 342 null values each  on all type of cluster columns:
#cluster_import,cluster_religious,cluster_community,cluster_highRisk,
#cluster_education,cluster_detentionCentre,cluster_workplace
X_train.isnull().sum()

#cases_new has 1 null
X_test.isnull().sum()

#Lets have the visual view 
#ploting missing values
X_train.isna().sum().plot(kind='bar')
plt.show

#Whereas, x_test has one null value at column cases_new
X_test.isnull().sum()

#checking on duplicate values
#no duplicated values for both X_train and X_test
X_train.duplicated().sum()
X_test.duplicated().sum()


#Looking at the large number of NaN is clusters columns 
#As all the information are equally important
#I will not drop any values
#Instead, I will replace missing values with median

#X_train data
#cluster_import
X_train[['cluster_import']] = X_train[['cluster_import']].fillna(X_train[['cluster_import']].median())

#cluster_religious
X_train[['cluster_religious']] = X_train[['cluster_religious']].fillna(X_train[['cluster_religious']].median())

#cluster_community
X_train[['cluster_community']] = X_train[['cluster_community']].fillna(X_train[['cluster_community']].median())

#cluster_highRisk
X_train[['cluster_highRisk']] = X_train[['cluster_highRisk']].fillna(X_train[['cluster_highRisk']].median())

#cluster_education
X_train[['cluster_education']] = X_train[['cluster_education']].fillna(X_train[['cluster_education']].median())

#cluster_detentionCentre
X_train[['cluster_detentionCentre']] = X_train[['cluster_detentionCentre']].fillna(X_train[['cluster_detentionCentre']].median())

#cluster_workplace
X_train[['cluster_workplace']] = X_train[['cluster_workplace']].fillna(X_train[['cluster_workplace']].median())


#cases_new
#firstly convert the empty values filled with nan then only find the median
X_train[['cases_new']] = X_train[['cases_new']].apply(pd.to_numeric,errors='coerce')
X_train[['cases_new']] = X_train[['cases_new']].fillna(X_train[['cases_new']].median())

#X_test data
#cases_new
X_test[['cases_new']] = X_test[['cases_new']].fillna(X_test[['cases_new']].median())

#The date is not required so will be dropping it in both X_train & X_test
X_train.drop('date', axis=1, inplace=True)
X_test.drop('date', axis=1, inplace=True)


#step 4: Data cleaning
#step 5: Feature selection

#step 6: Data preprocessing
mms = MinMaxScaler()
x_train  = X_train['cases_new'].values
x_train_scaled = mms.fit_transform(np.expand_dims(x_train, -1))

x_test = X_test['cases_new']
x_test_scaled = mms.transform(np.expand_dims(x_test,-1))

x_train = []
y_train = []

#testing data with window size 30days
for i in range(30,len(X_train)):
    x_train.append(x_train_scaled[i-30:i,0])
    y_train.append(x_train_scaled[i,0])
    

x_train = np.array(x_train)
y_train = np.array(y_train)

#testing dataset
#the axis = 0 refering to row
dataset_totest = np.concatenate((x_train_scaled,x_test_scaled),axis=0)
#slicing the last 80days (it is 30 days + length of testset 100)
data = dataset_totest[-130:]

x_test = []
y_test = []

for i in range(30,130):
    x_test.append(data[i-30:i,0])
    y_test.append(data[i,0])
    
#converting everything to array
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = np.expand_dims(x_train,axis=-1)
x_test = np.expand_dims(x_test,axis=-1)
#%% Model Creation

log_dir = os.path.join(LOG_PATH, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

#calling tensorboard
tensorboard_callback = TensorBoard(log_dir=log_dir)

#early stoping callback
early_stopping = EarlyStopping(monitor='loss', patience=3)

model = Sequential()
model.add(LSTM(64,activation='tanh',return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()

#%% Model compile 

model.compile(optimizer='adam', loss='mse', metrics='mse')

hist = model.fit(x_train,y_train,epochs=100, batch_size=32, 
                 callbacks=[tensorboard_callback,early_stopping]) 

print(hist.history.keys())

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['mse'])
plt.show()

#%%Save model 
model.save('cases_new.h5')
#%% Save scaler
filename = 'scaler.pkl'
pickle.dump(mms, open(filename, 'wb'))
SCALER_PATH = os.path.join(os.getcwd(),'scaler.pkl')

#%% Model Deployment
predicted = []

for i in x_test:
    predicted.append(model.predict(np.expand_dims(i,axis=0)))

predicted = np.array(predicted)
#%% Model Analysis

plt.figure()
plt.plot(predicted.reshape(len(predicted),1))
plt.plot(y_test)
plt.legend(['Predicted','Actual'])
plt.show()

#%% Model Evaluation

y_true=y_test
y_predicted = predicted.reshape(len(predicted),1)
print('MAPE:',(mean_absolute_error(y_true, y_predicted)/sum(abs(y_true)))*100)


