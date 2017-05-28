"""

(...)
It is important that the data used for training undergoes the same preprocessing as the data
used when making predictions.

"""


__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'

from numpy import percentile
from numpy import max
from numpy import empty, zeros
from scipy.signal import resample

def preprocess(data, data_fps,fps=100., filter=40., verbosity=0, fps_threshold=.1):
    """
    Normalize calcium traces
    
    Adapted from https://github.com/lucastheis/c2s (written by Lucas Theis, lucas@theis.io).
    
    This function does three things:
        1. Remove any linear trends using robust linear regression.
        2. Normalize the range of the calcium trace by the 5th and 80th percentile.
        3. Change the sampling rate of the calcium trace.
    
    In this script, the first step is generally replaced by estimating and removing a baseline using
    a percentile filter (40 seconds seems like a good value for the percentile filter).
    
    @type  data: list
    @param data: list of dictionaries containing calcium/fluorescence traces
    
    @type  fps: float
    @param fps: desired sampling rate of signals
    
    @type  filter: float/none
    @param filter: percentile filter length in seconds
    
    @type  filter: float/None
    @param filter: number of seconds used in percentile filter
    
    @type  verbosity: int
    @param verbosity: if positive, print messages indicating progress
    
    @type  fps_threshold: float
    @param fps_threshold: only resample if sampling rate differs more than this
    
    @rtype: list
    @return: list of preprocessed recordings
    """
    if fps is not None and fps > 0. and abs(data_fps - fps) > fps_threshold:
        # number of samples after update of sampling rate
        num_samples = int(float(data[:,0].size) * fps / data_fps + .5)
    else:
        # don't change sampling rate
        num_samples = data[:,0].size
    dataX = zeros((num_samples,data.shape[1]))
    for k in range(data.shape[1]):
        if verbosity > 0:
            print('Preprocessing calcium trace {0}...'.format(k))
        # remove linear trend
        data[:,k] = 	data[:,k] - percentile_filter(data[:,k], window_length=int(data_fps * filter), perc=5)
        
        # normalize dispersion
        calcium05 = percentile(data[:,k], 5)
        calcium80 = percentile(data[:,k], 80)
        
        if calcium80 - calcium05 > 0.:
            data[:,k] = (data[:,k] - calcium05) / float(calcium80 - calcium05)
        
        if num_samples != data[:,k].size:
            # factor by which number of samples will actually be changed
            factor = num_samples / float(data[:,k].size)
            # resample calcium signal
            dataX[:,k] = resample(data[:,k].ravel(), num_samples).reshape(1, -1)
        else:
            # don't change sampling rate
            num_samples = data[:,k].size
            dataX[:,k] = data[:,k]
    data_fps = data_fps * factor
    return dataX, data_fps


def percentile_filter(x, window_length, perc=5):
	"""
	For each point in a signal, computes a percentile from a window surrounding it.

	@type  window_length: int
	@param window_length: length of window in bins

	@type  perc: int
	@param perc: which percentile to compute

	@rtype: ndarray
	@return: array of the same size as C{x} containing the percentiles
	"""

	shape = x.shape
	x = x.ravel()
	y = empty(x.size)
	d = window_length // 2 + 1

	for t in range(x.size):
		fr = max([t - d + 1, 0])
		to = t + d
		y[t] = percentile(x[fr:to], perc)

	return y.reshape(shape)

