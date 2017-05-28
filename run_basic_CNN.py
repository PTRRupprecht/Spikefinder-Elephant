
# PART I. Generate the model

exec(open("elephant/config_elephant.py").read())
exec(open("elephant/1_load_data.py").read())
exec(open("elephant/2_model.py").read())
exec(open("elephant/3_preprocessing.py").read())

# PART II. Train the model
# Duration: >10 hours on a CPU
# PART II can be skipped if a pre-trained model is used (load_weights = 1)
model.fit(X, Y, batch_size=batch_size, epochs=nr_of_epochs)	

# evaluate the model
exec(open("elephant/4_evaluate.py").read())


# PART III. Use the model
# now use the model to predict spikes, save predictions to csv-files

# 1 = use training dataset, 0 = use test dataset
training = 1
# 1 = use pretrained networks, 0 = use the network that has been trained in this session
load_weights = 1

exec(open("elephant/5_make_and_save_predictions.py").read())

