"""

Make predictions using a model.
Save the predictions to a csv file.

"""

import pandas as pd
import numpy as np

from elephant.utils import norm

if training == 1:
	nb_neurons = 174
	nb_datasets = 10
else:
	nb_neurons = 32
	nb_datasets = 5

for iii, n_dataset in enumerate(range(1,nb_datasets+1)):
	print n_dataset
	if training:
		x1 = pd.read_csv("spikefinder.train/%d.train.calcium.csv" % n_dataset)
	else:
		x1 = pd.read_csv("spikefinder.test/%d.test.calcium.csv" % n_dataset)
    # convert to numpy arrays
	x1 = x1.values
	# initialize prediction array
	x1_predict = np.empty((x1.shape[0],x1.shape[1]))
	x1_predict[:] = np.NaN
	# calculate predictions
	number_of_neurons = x1.shape[1]
	for neuron_index in range(0,number_of_neurons):
		print neuron_index
		x1x = x1[:,neuron_index]
		idx = ~np.isnan(x1x)
		calcium_traceX = norm(x1x[idx])
		# initialize the prediction vector
		x1_predict[idx,neuron_index] = 0
		XX = np.zeros( (calcium_traceX.shape[0]-windowsize,windowsize,1),dtype = np.float32)
		for jj in range(0,(calcium_traceX.shape[0]-windowsize)):
			XX[jj,:,0] = calcium_traceX[jj:(jj+windowsize)]
		if load_weights:
			model.load_weights("models/model1.h5")
		Ypredict = model.predict( XX,batch_size = 4096 )

		x1_predict[idx,neuron_index] = 0
		x1_predict[int(windowsize*before_frac):int(windowsize*before_frac+len(Ypredict)),neuron_index] = Ypredict[:,0]

	x1_predict_panda = pd.DataFrame(x1_predict)
	if training:
		x1_predict_panda.to_csv("spikefinder.train/predictions/%d.spikes.calcium.csv" % n_dataset,index=False)
	else:
		x1_predict_panda.to_csv("spikefinder.test/predictions/%d.spikes.calcium.csv" % n_dataset,index=False)

