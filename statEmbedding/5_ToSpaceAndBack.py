# -*- coding: utf-8 -*-


for training in range(0,2):
	# dataset sizes are lateron used to facilitate weighted training
	dataset_sizes = np.array((740777, 672625, 563966, 165048, 149976, 171687, 875247, 486090, 584103, 697437),dtype=float)
	dataset_fraction_to_take = np.min(dataset_sizes)/dataset_sizes
	
	
	Performance = np.zeros([10,1])
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
			y1 = pd.read_csv("spikefinder.train/%d.train.spikes.csv" % n_dataset)
			y1 = y1.values
		else:
			x1 = pd.read_csv("spikefinder.test/%d.test.calcium.csv" % n_dataset)
	    # convert to numpy arrays
		x1 = x1.values
		# initialize prediction array
		x1_predict = np.empty((x1.shape[0],x1.shape[1]))
		x1_predict[:] = np.NaN
		
		# retrain model based on embedding
		model.load_weights("models/model1.h5")
		if training:
			distances = distances_train[n_dataset-1,:]
		else:
			distances = distances_test[n_dataset-1,:]
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
				execfile("elephant/3_preprocessing.py")
				X = X[0:int(X.shape[0]*distancesX[kk]*dataset_fraction_to_take[kk]),:,:]
				Y = Y[0:int(Y.shape[0]*distancesX[kk]*dataset_fraction_to_take[kk]),:]
				XX0 = np.concatenate((XX0,X),axis=0)
				YY0 = np.concatenate((YY0,Y),axis=0)
		
			learning_rate = 0.0033
			model.optimizer.lr.assign(learning_rate)
			model.fit(XX0, YY0, batch_size=batch_size, epochs=1)
		
		# calculate predictions
		number_of_neurons = x1.shape[1]
		performanceX = np.zeros((number_of_neurons,1))
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
			Ypredict = model.predict( XX,batch_size = 4096 )
			
			if training:
				YYpredict = np.zeros([len(idx),1])
				YYpredict[int(windowsize*before_frac): (calcium_traceX.shape[0] - int(windowsize*after_frac))] = Ypredict
				Prediction = np.convolve(YYpredict[0:calcium_traceX.shape[0]].ravel(),np.ones(4), 'valid')[::4]
				GroundTruth = np.convolve(y1[0:calcium_traceX.shape[0],neuron_index].ravel(),np.ones(4), 'valid')[::4]
				performanceX[neuron_index] = np.corrcoef(Prediction,GroundTruth)[0,1]
	
			x1_predict[idx,neuron_index] = 0
			x1_predict[int(windowsize*before_frac):int(windowsize*before_frac+len(Ypredict)),neuron_index] = Ypredict[:,0]
			
		Performance[iii] = np.median(performanceX)
		x1_predict_panda = pd.DataFrame(x1_predict)
		if training:
			x1_predict_panda.to_csv("spikefinder.train/predictions/%d.spikes.calcium.csv" % n_dataset,index=False)			
		else:
			x1_predict_panda.to_csv("spikefinder.test/predictions/%d.spikes.calcium.csv" % n_dataset,index=False)


#for jj in range(0,10):
#	print np.median(Performance[DatasetIX == (jj+1)	])


