# Uses parameters from 'MetaParameterExtraction.m' and 'SingleNeuronCNNs.py' to generate embedding spaces via PCA
# This was done in Matlab mainly for plotting.

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from copy import deepcopy

#A = sio.loadmat('statEmbedding/Parameters174py.mat')
#B2 = A['Parameters174']

A = sio.loadmat('statEmbedding/Confusion.mat',variable_names=['Confusion','Confusion_later','DaSet','DaSet2'])
Confusion = A['Confusion']
Confusion_later = A['Confusion_later']
DaSet = A['DaSet'][:,0]
DaSet2 = A['DaSet2'][0]


# normalize values for each neuron by how well this neuron be predicted by others (maximum except self-prediction)
ConfusionX = Confusion_later;
for j in range(0,174):
    ConfusionX[:,j] = ConfusionX[:,j]/np.mean(ConfusionX[:,j])



## calculate predicting quality

# normalize values for each neuron by how well this neuron can be predicted (maximum except self-prediction)
CX = Confusion_later - np.multiply(np.eye(174),Confusion_later)
for k in range(0,CX.shape[0]):
    CX[k,:] = CX[k,:]/np.percentile(CX[k,:],98)
    

# find for each cell k the neurons that can be predicted very well;
# rank neurons according to how well they predict neurons compared to
# how well those neurons are predicted by other well neurons


predicting_quality = np.zeros((174,))
for k in range(0,CX.shape[0]):
    position = np.zeros((174,))
    for j in range(0,CX.shape[0]):
        XX = np.sort(CX[j,:])
        iX = np.argsort(CX[j,:])
        position[j] = np.where(iX == k)[0][0]
    position_sorted = np.sort(position)
    predicting_quality[k] = 1/(1+np.mean(position_sorted[0:10]))
        

#plt.figure(13)
#plt.plot(predicting_quality)

## select neurons to be used for PCA

goodindizes = np.where(predicting_quality>np.percentile(predicting_quality,5))[0]
DaSet2 = DaSet2[goodindizes]
DaSet = DaSet[goodindizes]


## perform PCA to embed the prediction matrix into a lower dimension


ConfusionXX = ConfusionX[goodindizes,:][:,goodindizes]
for j in range(1,11):
    for k in range(1,11):
        indizes1 = np.where(DaSet==j)[0]
        indizes2 = np.where(DaSet==k)[0]
        temp = ConfusionXX[indizes1,:]
        temp = np.mean(np.mean(temp[:,indizes2]))
        ConfusionXX[np.ix_(indizes1,indizes2)] = temp
        
#plt.figure(132)
#plt.imshow(ConfusionXX+np.transpose(ConfusionXX)/2)
#plt.figure(133)
#plt.imshow(ConfusionX[goodindizes,:][:,goodindizes])


pca = PCA(n_components=2)
pca.fit(np.transpose(ConfusionXX)+ConfusionXX)

Confusion_transformed = pca.transform(np.transpose(ConfusionXX)+ConfusionXX)

#plt.figure(31)
#plt.plot(Confusion_transformed[:,0])
#plt.plot(Confusion_transformed[:,1])


symbolsX = ('o', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
plt.figure(95533),
for k in range(0,10):
    indizes = np.where(DaSet==k+1)[0]
    plt.plot(np.mean(Confusion_transformed[indizes,0]),np.mean(Confusion_transformed[indizes,1]),marker=symbolsX[k])
    plt.annotate(str(k+1),xy=(np.mean(Confusion_transformed[indizes,0])+0.5,np.mean(Confusion_transformed[indizes,1])))



embedding_2D_predictions = np.zeros((10,2))
for k in range(0,10):
    indizes = np.where(DaSet==k+1)[0]
    embedding_2D_predictions[k,0:2] = np.mean(Confusion_transformed[indizes,0:2],axis=0)



#%% to be saved to a mat file
#embedding_2D = mappedX(:,1:4);

# embedding space of statistics of the timetraces
# from MetaParameterExtraction
A = sio.loadmat('statEmbedding/Parameters174py.mat',variable_names=['DasetS','Parameters174'])
DasetS = A['DasetS']
Parameters174 = A['Parameters174']

Parameters174 = Parameters174[:,goodindizes]
DasetS = DasetS[goodindizes]

ParametersXX = deepcopy(Parameters174)
for k in range(0,10):
    indizes = np.where(DasetS==k+1)[0]
    for j in range(0,18):
        ParametersXX[j,indizes] = np.mean(ParametersXX[j,indizes])

pca2 = PCA(n_components=2)
pca2.fit(np.transpose(ParametersXX))
P174_tf = pca2.transform(np.transpose(ParametersXX))
#plt.plot(P174_tf)

embedding_2D_stats = np.zeros((10,2))
for j in range(0,10):
    indizes = np.where(DasetS==j+1)[0]
    embedding_2D_stats[j,:] = np.median(P174_tf[indizes,0:2],axis=0);

plt.figure(31222);
for k in range(0,10):
    plt.plot(embedding_2D_stats[k,0],embedding_2D_stats[k,1],marker=symbolsX[k])
    plt.annotate(str(k+1),xy=(embedding_2D_stats[k,0]+0.5,embedding_2D_stats[k,1]))




A = sio.loadmat('statEmbedding/Parameters32py.mat',variable_names=['DasetS32','Parameters32'])
DasetS32 = A['DasetS32']
Parameters32 = A['Parameters32']

ParametersXX = Parameters32;
for k in range(0,5):
    indizes = np.where(DasetS32==k+1)[0]
    for j in range(0,18):
        ParametersXX[j,indizes] = np.mean(ParametersXX[j,indizes])


# project into PCA space defined by the 174-neuron space
P33_tf = pca2.transform(np.transpose(ParametersXX))

embedding_2D_stats33 = np.zeros((5,2))
for j in range(0,5):
    indizes = np.where(DasetS32==j+1)[0]
    embedding_2D_stats33[j,:] = np.median(P33_tf[indizes,0:2],axis=0)

plt.figure(31222);
for k in range(0,5):
    plt.plot(embedding_2D_stats33[k,0],embedding_2D_stats33[k,1],marker=symbolsX[k],markersize=10)
    plt.annotate(str(k+1),xy=(embedding_2D_stats33[k,0]-0.5,embedding_2D_stats33[k,1]))


# save everything to a mat-file
sio.savemat('statEmbedding/embedding_spacesX.mat',{'embedding_2D_predictions':embedding_2D_predictions,'embedding_2D_stats':embedding_2D_stats,'embedding_2D_stats33':embedding_2D_stats33,'predicting_quality':predicting_quality,'goodindizes':goodindizes,'DaSet2':DaSet2,'DaSet':DaSet,'DasetS32':DasetS32})


