"""
Config file: collection of parameters to adjust learning and data pre-processing
"""


# determines how the time window used as input is positioned around the actual time point
before_frac, after_frac = 0.25, 0.75

# width of gaussian kernel used for convolution of spike train
spike_SD_kernel = 2.0 # half-size of kernel
spike_size_kernel = 15
# balanced set of non-spikes / spikes (not used as default)
balanced_labels = 0
verbosity = 1

# key: dataset number, value: list of neuron ids (zero-indexed)
# if value is None, use all neurons
datasets_to_train = {
	1: [0,1,2,3,4,5,6,7,8,9,10],
	2: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
	3: [0,1,2,3,4,5,6,7,8,9,10,11,12],
	4: [0,1,2,3,4,5],
	5: [0,1,2,3,4,5,6,7,8],
	6: [0,1,2,3,4,5,6,7,8],
	7: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36],
	8: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
	9: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
	10: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
}

spike_threshold = 0.1 # only used if spike_trace contains 'smooth' (only used with 'balanced_labels = 1')
spike_trace = "spikes_smooth" # "spikes"
calcium_trace = "calcium" # "calcium" # "calcium_smooth", "calcium_smooth_norm"

loss_function = 'mean_squared_error'
optimizer = 'Adagrad'
batch_size = 256
nr_of_epochs = 15

datasets_to_test = {
	1: [0,1,2,3,4,5,6,7,8,9,10],
	2: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
	3: [0,1,2,3,4,5,6,7,8,9,10,11,12],
	4: [0,1,2,3,4,5],
	5: [0,1,2,3,4,5,6,7,8],
	6: [0,1,2,3,4,5,6,7,8],
	7: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36],
	8: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
	9: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
	10: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
}


