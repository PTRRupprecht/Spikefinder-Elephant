#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo of how to use the program for deconvolving your data.

The demo is using a small dataset of 110 neurons, recorded simultaneously in a single FOV.
Recording rate: 28 Hz (resonant scanning).
Brain area: area Dp (piriform cortex homolog) and area Dl (hippocampal homolog) in adult zebrafish (brain explant, room temperature).
Calcium indicator: GCaMP6f, expressed using a NeuroD promotor fragment (https://www.osapublishing.org/boe/abstract.cfm?uri=boe-7-5-1656 for details).
Recording duration: ca. 3 minutes.
Only spontaneous activity.

@author: Peter Rupprecht, peter.rupprecht@fmi.ch
"""

from __future__ import print_function
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from elephant.utils2 import extract_stats, genhurst, map_between_spaces
from elephant.utils import norm
from elephant.c2s_preprocessing import preprocess, percentile_filter
from copy import deepcopy
from sklearn.decomposition import PCA
# PART 0. Preliminaries.


#_____Load data_____________________________________________________
# Load your calcium imaging data (modify according to your needs)

# load dataset as a simple 2D matrix, here using a mat file
trace = sio.loadmat('demo_dataset/Adult_zebrafish_110neurons.mat')['FFF']
# framerate used for ca imaging
fs = 28


#_____Preprocess____________________________________________________
# Preprocess data (upsample to 100 Hz, normalize offset/amplitude)
# The preprocessing is based on Theis et al., 2016
tracex, fsx = preprocess(trace,fs)



# PART I. The simple CNN model.


#_____Load CNN model_________________________________________________
exec(open("elephant/config_elephant.py").read())
exec(open("elephant/2_model.py").read())


#_____Load model weights_____________________________________________
model.load_weights("models/model1.h5")


#_____Make predictions_______________________________________________
Ypredict = np.zeros((tracex.shape[0]-windowsize,tracex.shape[1]))
for k in range(0,trace.shape[1]):
    print('Predicting spikes for neuron %s out of %s' % (k+1, trace.shape[1]))
    x1x = tracex[:,k]
    idx = ~np.isnan(x1x)
    calcium_traceX = norm(x1x[idx])
    # initialize the prediction vector
    XX = np.zeros( (calcium_traceX.shape[0]-windowsize,windowsize,1),dtype = np.float32)
    for jj in range(0,(calcium_traceX.shape[0]-windowsize)):
        XX[jj,:,0] = calcium_traceX[jj:(jj+windowsize)]
    A = model.predict( XX,batch_size = 4096 )
    Ypredict[idx[0:len(idx)-windowsize],k] = A[:,0]


#_____Plot  results__________________________________________________
plt.figure(101)
plt.imshow(np.transpose(Ypredict), aspect='auto')
plt.gray()
plt.clim(0.2,20)
plt.figure(102)
plt.imshow(np.transpose(tracex[int(windowsize*before_frac):calcium_traceX.shape[0]-int(windowsize*after_frac),:]), aspect='auto')
plt.jet()
plt.clim(0,8)



# PART II. The embedded model with focused re-training.


#_____Compute statistical porperties__________________________________
print('Compute statistical properties of your dataset ...')
stats = extract_stats(tracex[:,:])


#_____Compute location in embedding space_____________________________
A = sio.loadmat('statEmbedding/Parameters174py.mat',variable_names=['DasetS','Parameters174','Parameters174temp'])
DasetS = A['DasetS']
Parameters174temp = A['Parameters174temp']
Parameters174 = A['Parameters174']
goodindizes = sio.loadmat('statEmbedding/embedding_spacesX.mat')['goodindizes']

Parameters174 = Parameters174[:,goodindizes]
DasetS = DasetS[goodindizes]

ParametersXX = np.squeeze(deepcopy(Parameters174))
for k in range(0,10):
    indizes = np.where(DasetS==k+1)[0]
    for j in range(0,18):
        ParametersXX[j,indizes] = np.mean(ParametersXX[j,indizes])

pca2 = PCA(n_components=2)
pca2.fit(np.transpose(ParametersXX))

for k in range(0,18):
    stats[:,k] = (stats[:,k] - np.mean(Parameters174temp[:,k]))/np.std(Parameters174temp[:,k])
P1 = pca2.transform(stats)
P1mean = np.mean(P1,axis=0)
#plt.figure(31222)
#plt.plot(P1mean[0],P1mean[1],marker='d')


#_____Compute location in embedding space_____________________________
distances = map_between_spaces(P1mean)


#_____Retrain model in a local environment____________________________
exec(open("elephant/1_load_data.py").read())
# dataset sizes are lateron used to facilitate weighted training
dataset_sizes = np.array((740777, 672625, 563966, 165048, 149976, 171687, 875247, 486090, 584103, 697437),dtype=float)
dataset_fraction_to_take = np.min(dataset_sizes)/dataset_sizes

# importance of dataset decays with distance in embedding space
distancesX = np.exp(-(distances)/3.5)
distancesX = distancesX/np.max(distancesX)

# train for 5 epochs with changing datasubsets
for jjj in range(0,5):
    XX0 = np.empty((0,128,1))
    YY0 = np.empty((0,1))
    for kk in range(0,10):
        dataset_chosen = kk + 1
        datasets_to_train = {}
        IY = datasets == dataset_chosen
        datasets_to_train[dataset_chosen] = neurons[IY]
        verbosity = 0
        exec( open("elephant/3_preprocessing.py").read() )
        X = X[0:int(X.shape[0]*distancesX[kk]*dataset_fraction_to_take[kk]),:,:]
        Y = Y[0:int(Y.shape[0]*distancesX[kk]*dataset_fraction_to_take[kk]),:]
        XX0 = np.concatenate((XX0,X),axis=0)
        YY0 = np.concatenate((YY0,Y),axis=0)
        
    learning_rate = 0.0033
    model.optimizer.lr.assign(learning_rate)
    model.fit(XX0, YY0, batch_size=batch_size, epochs=1)



#_____Make refined predictions________________________________________
Ypredict = np.zeros((tracex.shape[0]-windowsize,tracex.shape[1]))
for k in range(0,trace.shape[1]):
    print('Predicting spikes for neuron %s out of %s' % (k+1, trace.shape[1]))
    x1x = tracex[:,k]
    idx = ~np.isnan(x1x)
    calcium_traceX = norm(x1x[idx])
    # initialize the prediction vector
    XX = np.zeros( (calcium_traceX.shape[0]-windowsize,windowsize,1),dtype = np.float32)
    for jj in range(0,(calcium_traceX.shape[0]-windowsize)):
        XX[jj,:,0] = calcium_traceX[jj:(jj+windowsize)]
    A = model.predict( XX,batch_size = 4096 )
    Ypredict[idx[0:len(idx)-windowsize],k] = A[:,0]


#_____Plot refined results______________________________________________
plt.figure(103)
plt.imshow(np.transpose(Ypredict), aspect='auto')
plt.gray()
plt.clim(0.2,20)
plt.figure(104)
plt.imshow(np.transpose(tracex[int(windowsize*before_frac):calcium_traceX.shape[0]-int(windowsize*after_frac),:]), aspect='auto')
plt.jet()
plt.clim(0,8)
