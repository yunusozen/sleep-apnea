import numpy as np
import pandas as pd
import os
import sklearn
import keras
import keras.backend as K
import seaborn as sn
import matplotlib.pyplot as plt
import h5py
import time
import datetime

from datetime import date

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from scipy.stats import norm

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

# name of model for tensorboard/saving
name = 'model-' + str(date.today()) + '-' + str(time.localtime().tm_hour) + '-' + str(time.localtime().tm_min)
print(name)

def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# change to current OS
operatingSystem = 'windows'

if operatingSystem is 'linux' or operatingSystem is 'macOS':
    inputsPath = '/data/inputs/'
    targetsPath = '/data/targets/'
else:
    inputsPath = '\\data\\inputs\\'
    targetsPath = '\\data\\targets\\'

# load a list of files into a 3D array of [samples, timesteps, features]
xLoaded = list()
yLoaded = []

print("Loading Data...")

for root, dirs, files in os.walk('.' + inputsPath):
    for fileName in files:
        xData = load_file(os.getcwd() + inputsPath + fileName)
        xLoaded.append(xData)
        yData = load_file(os.getcwd() + targetsPath + fileName)
        yLoaded.append(yData)

# stack group so that features are the 3rd dimension
X = np.stack(xLoaded, axis=0)
# Y is simply an array of data
Y = yLoaded

# use to check the balance of classes in the data
ones = 0
for event in Y:
    #print(event)
    if event == 1:
        ones += 1

print(((ones/len(Y))*100), "%")

Y = np.array(Y)

Y = to_categorical(Y)
Y = Y.reshape(-1, 2)

xShuffle, yShuffle = shuffle(X, Y, random_state=2)

print(X.shape)
print(Y.shape)

xTrain, xTest, yTrain, yTest = train_test_split(xShuffle, yShuffle, test_size=0.2)

print("Data Ready")

# def dPrime(y_true, y_pred):
#     matrix = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
#     tp, fn, fp, tn = matrix.ravel()
#     dPrime = norm.ppf(tp/(tp+fn)) - norm.ppf(tn/(tn+fp))
#     return K.constant(dPrime)

verbose, epochs, batch_size = 1, 30, 32
#CNN layers
model = Sequential()

model.add(Conv1D(filters=20, kernel_size=125, input_shape=(7500,1)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Dropout(0.3))

model.add(Conv1D(filters=40, kernel_size=50))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Dropout(0.3))

model.add(Conv1D(filters=60, kernel_size=10))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(25, activation='elu')) 
model.add(Dropout(0.3))

model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# initialise callbacks
es = EarlyStopping(monitor='val_acc', mode='max', patience=5, verbose=1, restore_best_weights=True)
tb = TensorBoard(log_dir='logs\\' + name, histogram_freq=1, write_graph=True, write_grads=True, write_images=True)

history = model.fit(xTrain, yTrain, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.1,
                    callbacks=[es,tb])
_, accuracy = model.evaluate(xTest, yTest, batch_size=batch_size, verbose=0)
yPred = model.predict(xTest)

# calculate accuracy as a percentage
accuracy = accuracy * 100.0
print('Accuracy =', accuracy, "%")

# generate confusion matrix
matrix = confusion_matrix(yTest.argmax(axis=1), yPred.argmax(axis=1))
print('Confusion Matrix:')
print(np.matrix(matrix))

# Calculate d' from testing
tp, fn, fp, tn = matrix.ravel()
dprime = norm.ppf(tp/(tp+fn)) - norm.ppf(tn/(tn+fp))
print('dPrime =', dprime)

# generate classification report
target_names = ['non-apnea', 'apnea']
print('Classification Report:')
print(classification_report(yTest.argmax(axis=1), yPred.argmax(axis=1), target_names=target_names))

# access the accuracy and loss values found throughout training
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
# d = history.history[dPrime]

epochs = range(1, len(acc) + 1)

# plot accuracy throughout training
plt.plot(epochs, acc, 'b')
plt.plot(epochs, val_acc, 'g')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.figure()

# plot loss throughout training
plt.plot(epochs, loss, 'b')
plt.plot(epochs, val_loss, 'g')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# plt.plot(epochs, d, 'bo')
# plt.title('Training d prime')
# plt.xlabel('Training Epochs')
# plt.ylabel('d prime')
# plt.show()

# save the model
model.save(name + '.h5')
print('Model Saved')