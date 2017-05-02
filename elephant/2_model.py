""""

Define the model using the function API of Keras.

"""

from keras.models import Sequential, Model
from keras.layers import  Dense, Dropout, Flatten, MaxPooling1D, Conv1D, Input
from keras.utils import np_utils
from keras import metrics
import os

windowsize = 128
conv_filter = Conv1D
write_file = True
preprocessing_done = False

filter_size =  (41, 21, 7)
filter_number = (50,60,70)
dense_expansion = 300	

	
if verbosity:
	print("setup training data with {} datasets".format(len(datasets_to_train) ))

inputs = Input(shape=(windowsize,1))

outX = conv_filter(filter_number[0], filter_size[0], strides=1, activation='relu')(inputs)
outX = conv_filter(filter_number[1], filter_size[1], activation='relu')(outX)
outX = MaxPooling1D(2)(outX)
outX = conv_filter(filter_number[2], filter_size[2], activation='relu')(outX)
outX = MaxPooling1D(2)(outX)

outX = Dense(dense_expansion, activation='relu')(outX)
outX = Flatten()(outX)
predictions = Dense(1,activation='linear')(outX)
model = Model(inputs=[inputs],outputs=predictions)
model.compile(loss=loss_function, optimizer=optimizer)


