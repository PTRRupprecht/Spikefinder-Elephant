"""
Uses a dictionary with datasets/neurons (datasets_to_train). Creates a large
matrix X that contains for each timepoint of each calcium trace a vector of
length 'windowsize' around the timepoint. The window is not symmetric around
the timepoint, but 75% into the future and 25% into the past.
Also creates a vector Y that contains the corresponding spikes/non-spikes.
Random permutations un-do the original sequence of the timepoints.

"""

import numpy as np

if verbosity:
	print("-----------------------------")
	print("setup training data with {} datasets".format(len(datasets_to_train) ))
totalX = []
totalY = []
for ds_idx, neuron_list in datasets_to_train.items():
	# for a single dataset, gather all training data
	if verbosity:
		print("process dataset", ds_idx)
	cntspike = 0
	cntnonspike = 0
	neuron_idx_to_train = {}
	if neuron_list is None:
		neuron_list = alldata2[ds_idx].keys()
	for neuron_idx in neuron_list:
		if np.any(neurons[datasets==ds_idx] == neuron_idx):
			if verbosity:
				print("process neuron", neuron_idx)
			d = alldata2[ds_idx][neuron_idx]
	
			if "smooth" in spike_trace:
				spiketimes = np.where( alldata2[ds_idx][neuron_idx][ spike_trace ] > spike_threshold )[0]
				nonspiketimes = np.where( alldata2[ds_idx][neuron_idx][ spike_trace ] <= spike_threshold )[0]
			else:
				spiketimes = np.where( alldata2[ds_idx][neuron_idx][ spike_trace] > 0 )[0]
				nonspiketimes = np.where( alldata2[ds_idx][neuron_idx][ spike_trace ] == 0 )[0]
	
			nr_of_timespoints = len(alldata2[ds_idx][neuron_idx][ spike_trace ])
			if verbosity:
				print("number of timepoints", nr_of_timespoints)
				print("number of spikes", len(spiketimes), " fraction ", 1.*len(spiketimes)/nr_of_timespoints)
				print("number of non spike bins", len(nonspiketimes), " fraction ", 1.*len(nonspiketimes)/nr_of_timespoints)
	
			np.random.shuffle( nonspiketimes )
			# this option would allow to balance the number of spike vs. non-spike bins (default off)
			if balanced_labels == 1:
				nonspiketimes_selected = nonspiketimes[:len(spiketimes)]
			else:
				nonspiketimes_selected = nonspiketimes[:]
			if verbosity:
				print("using {} fraction of nonspike windows, totalling {}, spike window count is {}".format( 1.*len(nonspiketimes_selected) / len(nonspiketimes), \
			 len(nonspiketimes_selected), len(spiketimes)) )
	
			neuron_idx_to_train[ neuron_idx] = {
				"spiketimes": spiketimes,
				"nonspiketimes": nonspiketimes_selected,
				"nr_of_timespoints": nr_of_timespoints
			}
			cntspike += len(neuron_idx_to_train[ neuron_idx]["spiketimes"])
			cntnonspike += len(neuron_idx_to_train[ neuron_idx]["nonspiketimes"])

	if verbosity:
		print("total number of spike times", cntspike, " and nonspike times", cntnonspike," from {} neurons".format(len(neuron_idx_to_train[neuron_idx])))
		print("ratio spike:nonspike", 1.*cntspike/cntnonspike)
	
		print("gather windows with spike from all spikes")
	Xwithspikes = np.zeros( (cntspike, windowsize, 1), dtype = np.float32 )
	Ywithspikes = np.zeros( (cntspike, 1), dtype = np.float32 )
	Xwithoutspikes = np.zeros( (cntnonspike, windowsize, 1), dtype = np.float32 )
	Ywithoutspikes = np.zeros( (cntnonspike, 1), dtype = np.float32 )

	i = 0
	j = 0
	for neuron_idx in neuron_list:
		if np.any(neurons[datasets==ds_idx] == neuron_idx):
			if verbosity:
				print("process neuron", neuron_idx)
			d = alldata2[ds_idx][neuron_idx]
			for idx in neuron_idx_to_train[neuron_idx]["spiketimes"]:
				start_idx = idx-int(windowsize*before_frac)
				end_idx = idx+int(windowsize*after_frac)
				if start_idx < 0 or end_idx > neuron_idx_to_train[neuron_idx]["nr_of_timespoints"] - windowsize:
					 continue
				Xwithspikes[i,:,0] = alldata2[ds_idx][neuron_idx][ calcium_trace ][start_idx:end_idx]
				Ywithspikes[i,:] = alldata2[ds_idx][neuron_idx][ spike_trace ][idx]
				i += 1
	
			for idx in neuron_idx_to_train[neuron_idx]["nonspiketimes"]:
				start_idx = idx-int(windowsize*before_frac)
				end_idx = idx+int(windowsize*after_frac)
				if start_idx < 0 or end_idx > neuron_idx_to_train[neuron_idx]["nr_of_timespoints"] - windowsize:
					continue
				Xwithoutspikes[j,:,0] = alldata2[ds_idx][neuron_idx][ calcium_trace ][start_idx:end_idx]
				Ywithoutspikes[j,:] = alldata2[ds_idx][neuron_idx][ spike_trace ][idx]
				j += 1

	# Concatenate and shuffle
	X = np.concatenate( [Xwithspikes[:i,:,:], Xwithoutspikes[:j,:,:]], axis = 0)
	Y = np.concatenate( [Ywithspikes[:i,:], Ywithoutspikes[:j,:]], axis = 0)
	p = np.random.permutation(len(X))
	X = X[p,:,:]
	Y = Y[p,:]

	totalX.append( X )
	totalY.append( Y )

X = np.concatenate( totalX, axis = 0)
Y = np.concatenate( totalY, axis = 0)
p = np.random.permutation(len(X))
X = X[p,:,:]
Y = Y[p,:]
