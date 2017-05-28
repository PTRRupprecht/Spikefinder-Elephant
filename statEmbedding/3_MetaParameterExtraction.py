# Code written by Peter Rupprecht (2017), ptrrupprecht.wordpress.com
# the code calculates statistical parameters for each neuron and saves it to disk


import numpy as np
import scipy.io as sio
from copy import deepcopy
from elephant.utils2 import extract_stats

## compute statistical properties of training dataset
DasetS = np.zeros((174,1))
Parameters174 = np.zeros((174,18))
counter = 0
for iii, n_dataset in enumerate(range(1,11)):
    print('Compute statistical properties of training dataset ',n_dataset)
    A = extract_stats("spikefinder.train/%d.train.calcium.csv" % n_dataset)
    DasetS[counter:counter+A.shape[0]] = n_dataset
    Parameters174[counter:counter+A.shape[0],:] = A
    counter += A.shape[0]

Parameters174temp = deepcopy(Parameters174)

for k in range(0,18):
    Parameters174[:,k] = (Parameters174[:,k] - np.mean(Parameters174temp[:,k]))/np.std(Parameters174temp[:,k])

sio.savemat('statEmbedding/Parameters174py.mat',{'Parameters174temp':Parameters174temp,'Parameters174':np.transpose(Parameters174),'DasetS':DasetS});



## compute statistical properties of test dataset
DasetS32 = np.zeros((32,1))
Parameters32 = np.zeros((32,18))
counter = 0
for iii, n_dataset in enumerate(range(1,6)):
    print('Compute statistical properties of test dataset ',n_dataset)
    A = extract_stats("spikefinder.test/%d.test.calcium.csv" % n_dataset)
    DasetS32[counter:counter+A.shape[0]] = n_dataset
    Parameters32[counter:counter+A.shape[0],:] = A
    counter += A.shape[0]

for k in range(0,18):
    Parameters32[:,k] = (Parameters32[:,k] - np.mean(Parameters174temp[:,k]))/np.std(Parameters174temp[:,k])

sio.savemat('statEmbedding/Parameters32py.mat',{'Parameters32':np.transpose(Parameters32),'DasetS32':DasetS32});



