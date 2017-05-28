import pandas as pd
import numpy as np

from elephant.utils import smooth, norm
								
# discard neuron 130 (spikes and calcium uncorrelated)
neurons = np.delete(neurons, 129, 0)
datasets = np.delete(datasets, 129, 0)

alldata2 = {}
for i in range(len(neurons)):
    dataset_index = int(datasets[i])
    neuron_index = int(neurons[i])
    if not dataset_index in alldata2:
        alldata2[ dataset_index ] = {}
    if not neuron_index in alldata2[ dataset_index ]:
        alldata2[ dataset_index ][ neuron_index ] = {"spikes": None, "calcium": None}

# kernel for smoothing the ground truth (spikes) to facilitate gradient descent
kernelX = np.exp(-(np.array(range(-spike_size_kernel,spike_size_kernel+1),'float32'))**2/spike_SD_kernel**2)
kernelX = kernelX/np.sum(kernelX)

for iii, n_dataset in enumerate(range(1,11)):
    # load using pandas
    x1 = pd.read_csv("spikefinder.train/%d.train.calcium.csv" % n_dataset)
    y1 = pd.read_csv("spikefinder.train/%d.train.spikes.csv" % n_dataset)
    # convert to numpy arrays
    x1 = x1.values
    y1 = y1.values
    number_of_neurons = x1.shape[1]
    for neuron_index in range(number_of_neurons):
        y1x = y1[:,neuron_index]
        x1x = x1[:,neuron_index]
        # discard NaNs
        idx = ~np.isnan(x1x)
        if np.any(neurons[datasets == n_dataset] == neuron_index):
            alldata2[ n_dataset ][ neuron_index ][ "spikes" ] = y1x[idx]
            alldata2[ n_dataset ][ neuron_index ][ "calcium" ] = norm(x1x[idx])
            alldata2[ n_dataset ][ neuron_index ][ "spikes_smooth" ] = norm(np.convolve(y1x[idx], kernelX, mode="same"))
#			alldata2[ n_dataset ][ neuron_index ][ "calcium_smooth" ] = smooth(x1x[idx], window_len=calcium_smoothing_windowsize)
#			alldata2[ n_dataset ][ neuron_index ][ "calcium_smooth_norm" ] = norm(smooth(x1x[idx], window_len=calcium_smoothing_windowsize))

print("DONE")
