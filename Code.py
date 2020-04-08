# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:30:23 2019

@author: HP
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np


import numpy as np
#importing data as a 2-D list.
import csv
data = [];
with open('test1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        data.append(row)
dataset = np.zeros((32,401,12));
y_ = np.zeros(32);
for i in range(len(data)):
    j = int(i/401);
    k = i%401;
    dataset[j][k][:] = data[i][:12];

y_train = [10, 20, 30, 40, 50, 60, 5, 15, 25, 35, 45, 55, 65, 70, 75, 80, 85, 90, 95, 12, 22, 32, 72, 82, 18, 28, 38, 48, 58, 78, 88, 89];


import numpy as np
#importing data as a 2-D list.
import csv
data = [];
with open('test2.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        data.append(row)
dataset_ = np.zeros((7,401,12));
y__ = np.zeros(7);
for i in range(len(data)):
    j = int(i/401);
    k = i%401;
    dataset_[j][k][:] = data[i][:12];
    y__[j] = data[j][12];

y_test = [42, 52, 62, 92, 18, 28, 68];





batch_size = 128
num_classes = 100
epochs = 1000

# input image dimensions
img_rows, img_cols = 401, 12

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = dataset.reshape(32,401,12,1)
x_test = dataset_.reshape(7,401,12,1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#from keras.utils import to_categorical
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(401,12,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))





model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor(random_state = 0)  
regressor.fit(x_train, y_train) 













import numpy as np
import csv
data = [];
k = 0;
with open('DATA2.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        data.append(row)
X_ = np.zeros(370);
y_ = np.zeros(37);
for i in range(len(data)-1):
    X_[i] = data[i][0];
    if(i%10 == 0):
        y_[k] = data[i][1];
        k=k+1;


X = np.reshape(X_, (-1, 10))


X_ = np.zeros((1224,10));
y__ = np.zeros(1224);
k = 0;
for i in range(36):
    for j in range(1,35):
        y__[k] = (y_[i]+y_[j])/2; 
        for l in range(10):
            X_[k][l] = (X[i][l]+X[j][l])/2 - X[0][l];
        k=k+1;
                
"""
A = np.zeros((1224,11));
for i in range(1224):
    A[i][:10] = X_[i][:];
    A[i][10] = y__[i];

from pandas import DataFrame
df = DataFrame(A)
export_excel = df.to_excel (r'DATAFinal.xlsx', index = None, header=True) 
"""





#Applying KNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_,y__, test_size=0.2, random_state=0)

knn = neighbors.KNeighborsRegressor(5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

sum(abs(y_test - y_pred))/len(y_test)




from sklearn.svm import SVR
clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
clf.fit(X_train, y_train);
y_pred = clf.predict(X_test)
clf.score(X_test, y_test)








#Applying ANN
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor


NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = 10, activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


NN_model.fit(X_, y__, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

pred = np.zeros(245);
pred = NN_model.predict(X_test)
a = 0;
for i in range(245):
    a+=abs(pred[i] - y_test[i]);
    print(abs(pred[i] - y_test[i]))





#LSTM
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



# design network
model = Sequential()
model.add(LSTM(50, input_shape=(10,1)))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(X_train, y_train, epochs=500, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

X__ = X_.reshape(1224,10,1);
X_test = X_test.reshape(245,10,1);
X_train = X_train.reshape(979,10,1); 

# make a prediction
yhat_ = model.predict(X_test)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)








