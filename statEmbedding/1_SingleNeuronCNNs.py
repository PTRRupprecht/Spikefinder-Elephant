# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:05:42 2017

Fits a small network with few parameters for each neuron.
Saves the model weights to disk.
Determines how well each model predicts the spikes for each neuron of the test dataset.

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.io as sio

from keras.models import Sequential, Model
from keras.layers import  Dense, Dropout, Flatten, MaxPooling1D, Conv1D, GlobalAveragePooling1D, Activation, LocallyConnected1D, Input, concatenate
from keras.utils import np_utils
from keras import metrics

### Fit models for each single neuron
print('Fit and save models for each single neuron ... this can take some time')
for iii in range(0,10):
    print('Dataset %s out of %s' % (iii+1, 10))
    n_dataset = iii + 1
    windowsize = 100;
    windowsize2 = 32;
    # load using pandas
    x1 = pd.read_csv("spikefinder.train/%d.train.calcium.csv" % n_dataset)
    y1 = pd.read_csv("spikefinder.train/%d.train.spikes.csv" % n_dataset)
    # convert to numpy arrays
    y1 = y1.values
    x1 = x1.values
    # discard NaNs
    
    for kkk in range(0,x1.shape[1]):
        n_neuron = kkk
        
        y1x = y1[:,n_neuron]
        x1x = x1[:,n_neuron]
        idx = ~np.isnan(x1x)
        y1x = y1x[idx]
        x1x = x1x[idx]
        
        # smooth the spikes in time
        SD_kernel = 2.5 # width of gaussian kernel used for convolution
        size_kernel = 15 # half-size of kernel
        kernelX = np.exp(-(np.array(range(-size_kernel,size_kernel+1),'float32'))**2/SD_kernel**2)
        kernelX = kernelX/np.sum(kernelX)    
        y1x0 = y1x
        y1x = np.convolve(y1x,kernelX,mode="same")
        #y1x = np.convolve(y1x.ravel(), np.ones(4), 'same')
								
        # normalization
        y1x = (y1x - np.mean(y1x))/np.std(y1x)
        x1x = (x1x - np.mean(x1x))/np.std(x1x)
        
        X = np.empty([x1x.shape[0]-windowsize, windowsize])
        for kk in range(0,x1x.shape[0]-windowsize):
            X[kk,:] = x1x[kk:kk+windowsize]
        
        # use only the valid part
        yX = y1x[int(windowsize/10*4):int(x1x.shape[0]-windowsize/10*6)]
        y1x0 = y1x0[int(windowsize/10*4):int(x1x.shape[0]-windowsize/10*6)]
        
        # process the data to fit into a keras NN
        XX = X.reshape((X.shape[0],windowsize,1))
        
        p = np.random.permutation(len(yX))
        XX = XX[p,:,:]
        yX = yX[p]     
        
        inputs = Input(shape=(windowsize,1))
        outX = LocallyConnected1D(32, 32,strides=2, activation='relu')(inputs)
        outX = Dropout(0.5)(outX)
        outX = LocallyConnected1D(16, 32, activation='relu')(outX)
        outX = MaxPooling1D(2)(outX)
        outX = Dropout(0.5)(outX)
        outX_hold = Dense(32, activation='relu')(outX)
        outX_hold = Flatten()(outX_hold)
        outX = Dropout(0.5)(outX_hold)
        predictions = Dense(1,activation='linear')(outX)
        
        model = Model(inputs=inputs,outputs=predictions)
        
        model.compile(loss='mean_squared_error',
              optimizer='Adagrad')
														
        ######### training ############################################################
        
        model.fit(XX, yX, batch_size=1024, epochs=5,verbose=2);
        model.save_weights('statEmbedding/single_cell_models/my_model_weights%d_%s.h5' % (n_dataset,n_neuron) )

        model.fit(XX, yX, batch_size=1024, epochs=10,verbose=2);
        model.save_weights('statEmbedding/single_cell_models/my_model_later%d_%s.h5' % (n_dataset,n_neuron) )

### Apply every model (indices ii,jj) to each single neuron (iii,jjj)
print('Apply every model to each single neuron ... this can take some time')

BBX = np.zeros([10,37,10,37])
BBX_later = np.zeros([10,37,10,37])
performance = np.zeros([10,37])
for iii in range(0,10):
    # select a single neuron of a single dataset 
#    windowsize = windowsize1;
    n_dataset = iii+1
    x1 = pd.read_csv("spikefinder.train/%d.train.calcium.csv" % n_dataset)
    y1 = pd.read_csv("spikefinder.train/%d.train.spikes.csv" % n_dataset)
    # convert to numpy arrays
    y1 = y1.values
    x1 = x1.values
    # discard NaNs
    print('Processing dataset %s out of %s' % (n_dataset, 10))
    for jjj in range(0,x1.shape[1]):
        n_neuron = jjj
        y1x = y1[:,n_neuron]
        x1x = x1[:,n_neuron]
        # discard NaNs
        idx = ~np.isnan(x1x)
        y1x = y1x[idx]
        x1x = x1x[idx]  
        
        # smooth the spikes in time
        size_kernel = 15 # half-size of kernel
        kernelX = np.exp(-(np.array(range(-size_kernel,size_kernel+1),'float32'))**2/SD_kernel**2)
        kernelX = kernelX/np.sum(kernelX)    
        y1x0 = y1x
        y1x = np.convolve(y1x,kernelX,mode="same")
        
        y1x = (y1x - np.mean(y1x))/np.std(y1x)
        x1x = (x1x - np.mean(x1x))/np.std(x1x)

        X = np.empty([x1x.shape[0]-windowsize, windowsize])
        for kk in range(0,x1x.shape[0]-windowsize):
            X[kk,:] = x1x[kk:kk+windowsize]
    
        # use only part of the target 
        y1x = y1x[int(windowsize/10*4):int(x1x.shape[0]-windowsize/10*6)]
        y1x0 = y1x0[int(windowsize/10*4):int(x1x.shape[0]-windowsize/10*6)]
        
        # process the data to fit into a keras NN
        X = X.reshape((X.shape[0],windowsize,1))
        
        y = y1x #np_utils.to_categorical(y1x)    
        
        for ii in range(0,10):
            n_datasetX = ii+1
            x0x = pd.read_csv("spikefinder.train/%d.train.calcium.csv" % n_datasetX)
            for jj in range(0,x0x.shape[1]):
                n_neuronX = jj
                model.load_weights('statEmbedding/single_cell_models/my_model_weights%d_%s.h5' % (n_datasetX,n_neuronX) )
                y_pred = model.predict(X, batch_size=32, verbose=0)
        
                # calculate performance based on correlation as used with the spikefinder challenge (albeit with smoothed ground truth)
                Prediction = np.convolve(y_pred.ravel(),np.ones(4), 'valid')[::4]
                GroundTruth = np.convolve(y1x0.ravel(), np.ones(4), 'valid')[::4]
        
                BBX[iii,jjj,ii,jj] = np.corrcoef(Prediction,GroundTruth)[0,1]
                
                model.load_weights('statEmbedding/single_cell_models/my_model_later%d_%s.h5' % (n_datasetX,n_neuronX) )
                y_pred = model.predict(X, batch_size=32, verbose=0)
        
                # calculate performance based on correlation as used with the spikefinder challenge (albeit with smoothed ground truth)
                Prediction = np.convolve(y_pred.ravel(),np.ones(4), 'valid')[::4]
                GroundTruth = np.convolve(y1x0.ravel(), np.ones(4), 'valid')[::4]
        
                BBX_later[iii,jjj,ii,jj] = np.corrcoef(Prediction,GroundTruth)[0,1]


### save the data to a mat-file (Confusion matrix)
# Confusion: confusion matrix after 5 training epochs
# Confusion: confusion matrix after 15 training epochs
# DaSet: Dataset indices for each neuron
# DaSet2: Neuron indices for each neuron


BXX = BBX
counter = 0
DaSet = np.zeros([174,1],dtype='int')
DaSet2 = np.zeros([174],dtype='int')
for iii in range(0,10):
    n_dataset = iii + 1
    # load using pandas
    x1 = pd.read_csv("spikefinder.train/%d.train.calcium.csv" % n_dataset)
    # convert to numpy arrays
    x1 = x1.values

    DaSet[counter:(counter+x1.shape[1])] = int(n_dataset)
    DaSet2[counter:(counter+x1.shape[1])] = range(0,x1.shape[1])
    counter += x1.shape[1]

Confusion = np.zeros([174,174])
Confusion_later = np.zeros([174,174])
for jj in range(0,174):
    for kk in range(0,174):
        Confusion[jj,kk] = BXX[int(DaSet[jj]-1),DaSet2[jj],int(DaSet[kk]-1),DaSet2[kk]]
        Confusion_later[jj,kk] = BBX_later[int(DaSet[jj]-1),DaSet2[jj],int(DaSet[kk]-1),DaSet2[kk]]

plt.figure(33)
plt.clf
plt.imshow(Confusion, interpolation='none')
plt.figure(34)
plt.clf
plt.imshow(Confusion_later, interpolation='none')

sio.savemat('statEmbedding/Confusion.mat', {'Confusion':Confusion,'Confusion_later':Confusion_later,'DaSet':DaSet,'DaSet2':DaSet2})


