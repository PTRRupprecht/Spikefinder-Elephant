#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 03:34:06 2017

@author: Peter Rupprecht, peter.rupprecht@fmi.ch
"""

import numpy as np
import pandas as pd
import warnings

import scipy.io as sio
import scipy.stats as sistats

from scipy.signal import welch
from sklearn import tree

# documented version of this function to be found at https://github.com/PTRRupprecht/GenHurst
def genhurst(S,q):

    L=len(S)       
    if L < 100:
        warnings.warn('Data series very short!')
       
    H = np.zeros((len(range(5,20)),1))
    k = 0
    
    for Tmax in range(5,20):
        
        x = np.arange(1,Tmax+1,1)
        mcord = np.zeros((Tmax,1))
        
        for tt in range(1,Tmax+1):
            dV = S[np.arange(tt,L,tt)] - S[np.arange(tt,L,tt)-tt] 
            VV = S[np.arange(tt,L+tt,tt)-tt]
            N = len(dV) + 1
            X = np.arange(1,N+1)
            Y = VV
            mx = np.sum(X)/N
            SSxx = np.sum(X**2) - N*mx**2
            my = np.sum(Y)/N
            SSxy = np.sum( np.multiply(X,Y))  - N*mx*my
            cc1 = SSxy/SSxx
            cc2 = my - cc1*mx
            ddVd = dV - cc1
            VVVd = VV - np.multiply(cc1,np.arange(1,N+1)) - cc2
            mcord[tt-1] = np.mean( np.abs(ddVd)**q )/np.mean( np.abs(VVVd)**q )
            
        mx = np.mean(np.log10(x))
        SSxx = np.sum( np.log10(x)**2) - Tmax*mx**2
        my = np.mean(np.log10(mcord))
        SSxy = np.sum( np.multiply(np.log10(x),np.transpose(np.log10(mcord)))) - Tmax*mx*my
        H[k] = SSxy/SSxx
        k = k + 1
        
    mH = np.mean(H)/q
    
    return mH

# extract statistical properties of a set of preprocessed neuronal calcium imaging time series
def extract_stats(inputX):
    # check whether input is a filename or a matrix
    if isinstance(inputX, str):
        # load using pandas
        x1 = pd.read_csv(inputX)
        # convert to numpy array
        x1 = x1.values
    else:
        x1 = inputX
        if x1.shape[0] < x1.shape[1]:
            x1 = np.transpose(x1)
            
    number_of_neurons = x1.shape[1]
    
    
    Kurtouis = np.zeros((number_of_neurons,1))
    Skewness = np.zeros((number_of_neurons,1))
    Varianz = np.zeros((number_of_neurons,1))
    Corr2sec = np.zeros((number_of_neurons,1))
    Corr1sec = np.zeros((number_of_neurons,1))
    Corr0sec = np.zeros((number_of_neurons,1))
    HurstExp = np.zeros((number_of_neurons,5))
    NoiseFreq = np.zeros((number_of_neurons,7))
    
    counter = 0
    for neuron_index in range(number_of_neurons):
        x1x = x1[:,neuron_index]
        # discard NaNs
        idx = ~np.isnan(x1x)
        L_trace = x1x[idx]
        
        Varianz[counter] = np.var(L_trace)/np.mean(L_trace)
        L_trace = (L_trace-np.median(L_trace))/np.std(L_trace)
        Kurtouis[counter] = sistats.kurtosis(L_trace,fisher=False)
        Skewness[counter] = sistats.skew(L_trace)
        Corr0sec[counter],_ = sistats.pearsonr(L_trace,np.roll(L_trace,50))
        Corr1sec[counter],_ = sistats.pearsonr(L_trace,np.roll(L_trace,100))
        Corr2sec[counter],_ = sistats.pearsonr(L_trace,np.roll(L_trace,200))
        # Hurst exponents
        for k in range(0,5):
            HurstExp[counter,k] = genhurst(L_trace,k+1)
        
        # noise spectrum readout
        vect_freq_noise,PSD_noise = welch(L_trace,fs=100,window='hamming',nperseg=1024,noverlap=512,scaling='spectrum')
        # cut off timetrace at the actual sampling frequency
        max_freq = (np.abs(PSD_noise)<1e-4).tostring().decode().find('\x01')
        PSD_noise  = PSD_noise/np.sum(PSD_noise[0:max_freq-1])
        
        NoiseFreq[counter,:] = (-np.log(PSD_noise[np.arange(0,37,6)])-np.log(PSD_noise[np.arange(1,38,6)]))/2
        counter = counter + 1

    Parameters = np.column_stack( ( Varianz, Kurtouis, Skewness, Corr0sec, Corr1sec, Corr2sec,
                                   HurstExp[:,0] ,HurstExp[:,1] ,HurstExp[:,2] ,HurstExp[:,3] ,HurstExp[:,4], 
                                   NoiseFreq[:,0],NoiseFreq[:,1],NoiseFreq[:,2],NoiseFreq[:,3],NoiseFreq[:,4],NoiseFreq[:,5],NoiseFreq[:,6] ) )
            
    return Parameters


# Uses location in statistical embedding space.
# Computes the location in the predictive embedding space.
# Returns the distances of the computed location to the locations of the training datasets 1-10 in the predictive embedding space.
def map_between_spaces(embedding_space_location):
    
    embedding_space_location = embedding_space_location.reshape(1, -1)
    
    AA = sio.loadmat('statEmbedding/embedding_spacesX.mat')
    Embedding_crossmodel = np.array(AA['embedding_2D_predictions'])
    Embedding_stats = np.array(AA['embedding_2D_stats'])
    
    predictior = np.zeros(embedding_space_location.shape)
    for kk in range(0,150):
        clfs1 = tree.DecisionTreeRegressor(criterion='mae',max_depth=7)
        clfs2 = tree.DecisionTreeRegressor(criterion='mae',max_depth=7)
        predictiorX = np.zeros(embedding_space_location.shape)
        clfs1.fit(Embedding_stats, Embedding_crossmodel[:,0])
        clfs2.fit(Embedding_stats, Embedding_crossmodel[:,1])
        
        predictiorX[:,0] = clfs1.predict(embedding_space_location)
        predictiorX[:,1] = clfs2.predict(embedding_space_location)
        predictior = predictior + predictiorX
    predictior = predictior/150.0
        
    distances = np.zeros((10,1))
    for jj in range(0,10):
        distances[jj,0] = np.sqrt(np.sum( (predictior[0,:] - Embedding_crossmodel[jj,:])**2) )
    
    return distances