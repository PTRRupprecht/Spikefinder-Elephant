
# Parts I and II produce outputs that are also available as mat-files and can therefore be omitted.
# Especially Part I can take long (>1d on a CPU)

# PART I. Generate the confusion matrix (how well does neuron A predict spikes for neuron B?).
# The confusion matrix is saved to 'statEmbedding/Confusion.mat'.
exec(open("statEmbedding/1_SingleNeuronCNNs.py").read())

# PART II. Generate the embedding spaces for the predictive confusion matrix and for
# the statistical property similarity.
exec(open("statEmbedding/2_Generate_embedding_spaces.py").read())
exec(open("statEmbedding/3_MetaParameterExtraction.py").read())
# You can go ahead and simply use the result, which is saved to 'statEmbedding/embedding_spacesX.mat'.

# PART III. Config, Load data, define model, etc.
# Same scripts as in 'run_basic_CNN.py'
exec(open("elephant/config_elephant.py").read())
exec(open("elephant/1_load_data.py").read())
exec(open("elephant/2_model.py").read())

# PART IV. Generate a fit for the transformation between the embedding spaces.
# re-train the model on a focused subset, chosen to be close in embedding space
exec(open("statEmbedding/4_ToSpaceAndBack.py").read())

