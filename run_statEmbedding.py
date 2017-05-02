

# PART I. Generate the confusion matrix (how well does neuron A predict spikes for neuron B?).
# The confusion matrix is saved to 'statEmbedding/Confusion.mat'.
exec(open("statEmbedding/1_SingleNeuronCNNs.py").read())

# PART II. Generate the embedding spaces for the predictive confusion matrix and for
# the statistical property similarity.
# This is done in Matlab via 'MetaParameterExtraction.m' and 'Generate_embedding_spaces.m'.
# You can go ahead and simply use the result, which is saved to 'embedding_spaces.mat'.

# PART III. Generate a fit for the transformation between the embedding spaces.
# Important output are the variables 'distances_train' and 'distances_test'.
exec(open("statEmbedding/4_MappingBetweenSpaces.py").read())


# PART IV. Config, Load data, define model, etc.
exec(open("elephant/config_elephant.py").read())
exec(open("elephant/1_load_data.py").read())
exec(open("elephant/2_model.py").read())

# re-train the model on a focused subset, chosen to be close in embedding space
exec(open("statEmbedding/5_ToSpaceAndBack.py").read())

