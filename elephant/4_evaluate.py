#The script uses the neurons selected for the test dataset (datasets_to_test)
#to evaluate the performance of the model 'model'.
#
#The evaluation is based on the correlation coefficient of ground truth and
#prediction, both of them downsampled to 25 Hz, as for the evaluation via
#spikefinder.codeneuro.org.
#
#performance_metrics gives a 10-element vector with the median of the correlation
#coefficient, taken over all neurons of the respective dataset.


import numpy as np
from keras import metrics


alldata3 = {}
for i in range(len(neurons)):
    dataset_index = int(datasets[i])
    neuron_index = int(neurons[i])
    if not dataset_index in alldata3:
        alldata3[ dataset_index ] = {}
    if not neuron_index in alldata3[ dataset_index ]:
        alldata3[ dataset_index ][ neuron_index ] = {"prediction": None, "performance": None}

performance_metrics = np.zeros([10,1])
for ds_idx, neuron_list in datasets_to_test.items():
	# for a single dataset, gather all training data
	print("process dataset", ds_idx)
	Performance = np.zeros([len(neuron_list),1])
	for neuron_idx in neuron_list:
		d = alldata2[ds_idx][neuron_idx]
		calcium_data = d[ calcium_trace ]
		spike_gt = d["spikes"]
		number_of_points = len(calcium_data)
		Xtrain = np.zeros( (number_of_points, windowsize, 1), dtype=np.float32 )
		for idx in range(int(windowsize*before_frac), number_of_points - int(windowsize*after_frac)):
			start_idx = idx - int(windowsize*before_frac)
			end_idx = idx + int(windowsize*after_frac)
			Xtrain[idx,:,0] = calcium_data[start_idx:end_idx]
	
		Ypredict = model.predict( Xtrain )
		Ypredict[0:int(windowsize*before_frac)] = np.mean(Ypredict)
		Ypredict[ (number_of_points - int(windowsize*after_frac))::] = np.mean(Ypredict)
		Prediction = np.convolve(Ypredict.ravel(),np.ones(4), 'valid')[::4]
		GroundTruth = np.convolve(spike_gt.ravel(),np.ones(4), 'valid')[::4]
		Performance[neuron_idx] = np.corrcoef(Prediction,GroundTruth)[0,1]
		print("process neuron", neuron_idx,"Performance", int(100*Performance[neuron_idx]))
		alldata3[ ds_idx ][ neuron_idx ][ "prediction" ] = Ypredict
		alldata3[ ds_idx ][ neuron_idx ][ "performance" ] = Performance[neuron_idx]
	print("dataset", ds_idx," processed, Performance", int(100*np.median(Performance)))
	performance_metrics[ds_idx-1] = np.median(Performance)
