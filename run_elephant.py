#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# PART I. Generate the model

exec(open("elephant/config_elephant.py").read())
exec(open("elephant/1_load_data.py").read())
exec(open("elephant/2_model.py").read())
exec(open("elephant/3_preprocessing.py").read())

# train the model
model.fit(X, Y, batch_size=batch_size, epochs=nr_of_epochs)	

# evaluate the model
exec(open("elephant/4_evaluate.py").read())


# PART II. Use the model
# now use the model to predict spikes, save predictions to csv-files

# 1 = use training dataset, 0 = use test dataset
training = 1
# 1 = use pretrained networks, 0 = use the network that has been trained in this session
load_weights = 1

execfile("elephant/5_make_and_save_predictions.py")
