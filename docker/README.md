This docker container is based on Keras' Docker Image Build process

	https://github.com/fchollet/keras/tree/master/docker

In addition to Keras, this Dockerfile clones the Spikefinder-Elephant Repository to /elephant. You can start a jupyter notebook and then open the "Demo.ipynb" file after building the container:

	$ make notebook
