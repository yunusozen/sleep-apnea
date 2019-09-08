import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import os


model = load_model('models\\model-09-08-19-2.h5')

# look at model summary to see shapes between layers etc.
model.summary()


# summarize filter shapes
for layer in model.layers:
    # only want conv layers (for now)
    if 'conv' not in layer.name:
        continue
    # get filter weights
    filters, biases = layer.get_weights()
    print(layer.name, filters.shape)

# visualise the filters in the first layer (conv layers are layers 0, 5 and 10)
filters1, biases1 = model.layers[0].get_weights()
for i in range(20):
    f = filters1[:, :, i]
    plt.subplot(10,2,(i+1))
    plt.plot(f)
plt.figure()

# perform Fourier transforms on the filters in the first layer
# Fs = 250
# Ts = 1/Fs
# for i in range(20):
#     f = filters1[:, :, i]
#     y = fft(f) / len(f)
#     x = np.linspace(0, len(y), len(y))
#     yAmp = 2*np.abs(y[0:len(y)//2])
#     yAmp[0] = yAmp[0]/2
#     plt.subplot(10,2,(i+1))
#     plt.plot(x[0:len(yAmp)], yAmp)
#     plt.xlim(0,30)
# plt.figure()

# plot spectrograms of each filter
Fs = 250
mode = 'psd'
for i in range(20):
    f = filters1[:, :, i]
    plt.subplot(10,2,(i+1))
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(f, Fs=Fs, mode=mode)
    plt.ylim(0, 30)

# plot a specific filter and its spectrogram
filterNumber = 6    # change to the particular filter you want to visulaise

f = filters1[:, :, filterNumber]
fig, ax = plt.subplots()
plt.subplot(2,1,1)
plt.plot(f)
plt.xlabel('Points')
plt.ylabel('Weights')

plt.subplot(2,1,2)
powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(f, Fs=Fs, mode=mode)
plt.ylim(0, 30)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
#fig.colorbar(imageAxis)    # display colour bar (not sure if necesarry)
plt.figure()

filters2, biases2 = model.layers[5].get_weights()
for i in range(40):
    f = filters2[:, filterNumber, i]
    plt.subplot(10,4,(i+1))
    plt.plot(f)
plt.figure()

for i in range(40):
    f = filters2[:, filterNumber, i]
    plt.subplot(10,4,(i+1))
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(f, Fs=Fs, mode=mode)
    plt.ylim(0, 30)

plt.show()