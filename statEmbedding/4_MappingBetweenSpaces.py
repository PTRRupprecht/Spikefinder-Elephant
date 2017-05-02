
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


#### load 2D labels ####################################

AA = sio.loadmat('statEmbedding/embedding_spaces.mat')
Embedding_crossmodel = np.array(AA['embedding_2D_predictions'])
Embedding_stats = np.array(AA['embedding_2D_stats'])
Embedding_stats_test = np.array(AA['embedding_2D_stats33'])
Predict_quality = np.array(AA['predicting_quality'])
Good_neurons = np.array(AA['goodindizes'])
DatasetIX_good = np.array(AA['DaSet'])
NeuronIX_good = np.array(AA['DaSet2'])

#### load dataset in small junks, timetraces XX, labels YY ####

from sklearn.svm import SVR
from sklearn import tree, decomposition

predictior = np.zeros([10,2])
predictior_test = np.zeros([5,2])
for kk in range(0,150):
	clfs1 = tree.DecisionTreeRegressor(criterion='mae',max_depth=7)
	clfs2 = tree.DecisionTreeRegressor(criterion='mae',max_depth=7)
	predictiorX = np.zeros([10,2])
	predictiorX_test = np.zeros([5,2])
	clfs1.fit(Embedding_stats, Embedding_crossmodel[:,0])
	clfs2.fit(Embedding_stats, Embedding_crossmodel[:,1])
	
	predictiorX[:,0] = clfs1.predict(Embedding_stats)
	predictiorX[:,1] = clfs2.predict(Embedding_stats)
	predictiorX_test[:,0] = clfs1.predict(Embedding_stats_test)
	predictiorX_test[:,1] = clfs2.predict(Embedding_stats_test)
	predictior = predictior + predictiorX
	predictior_test = predictior_test + predictiorX_test
predictior_test = predictior_test/150.0
predictior = predictior/150.0

#plt.figure(42)
#plt.clf
#plt.plot( Embedding_crossmodel[:,0],predictior[:,0],'ro')
#plt.plot( Embedding_crossmodel[0:5,0],predictior_test[:,0],'ko')


distances_train = np.zeros((10,10))
for kk in range(0,10):
	for jj in range(0,10):
		distances_train[kk,jj] = np.sqrt(np.sum((predictior[kk,:] - Embedding_crossmodel[jj,:])**2))

distances_test = np.zeros((5,10))
for kk in range(0,5):
	for jj in range(0,10):
		distances_test[kk,jj] = np.sqrt(np.sum((predictior_test[kk,:] - Embedding_crossmodel[jj,:])**2))

#plt.figure(2)
#plt.clf
#plt.imshow(distances_train,interpolation='None')
