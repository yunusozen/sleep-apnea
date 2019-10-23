import numpy as np
import pandas as pd
import os
import sklearn
import keras
import keras.backend as K
import seaborn as sn
import matplotlib.pyplot as plt
import h5py

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

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

#change to current OS
operatingSystem = 'windows'

if operatingSystem == 'linux':
    inputsPath = '/data/inputs/'
    targetsPath = '/data/targets/'
else:
    inputsPath = '\\data\\inputs\\'
    targetsPath = '\\data\\targets\\'

# loads data from a single .csv file
def load_file(filepath):
	dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# Set random seed
seed = 7
np.random.seed(seed)

# load a list of files into a 3D array of [samples, timesteps, features]
xLoaded = list()
yLoaded = []

print("Loading Data...")

# load input and target .csv files
for root, dirs, files in os.walk('.' + inputsPath):
    for fileName in files:
        xData = load_file(os.getcwd() + inputsPath + fileName)
        xLoaded.append(xData)
        yData = load_file(os.getcwd() + targetsPath + fileName)
        yLoaded.append(yData)

# data loaded in
X = np.stack(xLoaded, axis = 0) 
Y = yLoaded
Y = np.array(Y)

# change targets to one hot encoded form
Y = to_categorical(Y)
Y = Y.reshape(-1, 2)

# create 10 splits within the data
kFold = KFold(n_splits=10, shuffle=True, random_state=seed)

# initialise lists for saving results
accuracyList = list()
cmList = list()
crList = list()
dpList = list()

verbose, epochs, batch_size = 0, 30, 32
fold = 1

for train, test in kFold.split(X, Y):
    print('Fold: ', fold)
    
    # Create model
    model = Sequential()

    # Convolutional Layer 2
    model.add(Conv1D(filters=20, kernel_size=125, input_shape=(7500,1)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=10))
    model.add(Dropout(0.3))

    # Convolutional Layer 2
    model.add(Conv1D(filters=40, kernel_size=50))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.3))

    # Convolutional Layer 3
    model.add(Conv1D(filters=60, kernel_size=10))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.3))

    # Fully Connected Layer 1
    model.add(Flatten())
    model.add(Dense(10, activation='elu')) 
    model.add(Dropout(0.3))

    # Fully Connected Layer 2
    model.add(Dense(2, activation='softmax'))

    # configure model for training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train model with early stopping call back
    es = EarlyStopping(monitor = 'val_acc', mode = 'max', patience = 5, verbose = 1, restore_best_weights = True)
    model.fit(X[train], Y[train], epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split = 0.1, callbacks=[es])

    # test model
    accuracy = model.evaluate(X[test], Y[test], batch_size=batch_size, verbose=0)
    yPred = model.predict(X[test])

    print('Accuracy =', accuracy[1]*100, "%")
    accuracyList.append(accuracy[1]*100)

    # generate confusion matrix
    matrix = confusion_matrix(Y[test].argmax(axis=1), yPred.argmax(axis=1))
    print('Confusion Matrix:')
    print(np.matrix(matrix))
    cmList.append(matrix)

    # calculate d'
    tp, fn, fp, tn = matrix.ravel()
    dprime = norm.ppf(tn/(tn+fp)) - norm.ppf(fn/(tp+fn))
    print('dPrime =', dprime)
    dpList.append(dprime)

    # generate classification report
    target_names = ['non-apnea', 'apnea']
    print('Classification Report:')
    cr = classification_report(Y[test].argmax(axis=1), yPred.argmax(axis=1), target_names=target_names)
    print(cr)
    crList.append(cr)

    # save each model
    modelName = 'model' + str(fold) + '.h5'
    model.save(modelName)

    fold = fold + 1

print('Mean accuracy = ', np.mean(accuracyList), 'Standard Deviation =', np.std(accuracyList))
print('Mean dPrime = ', np.mean(dpList), 'Standard Deviation =', np.std(dpList))

