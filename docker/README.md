This docker container is based on Keras' Docker Image Build process

	https://github.com/fchollet/keras/tree/master/docker

In addition to Keras, this Dockerfile clones the Spikefinder-Elephant Repository to /elephant.

For the GPU version, make sure that also the [Nvidia docker](https://github.com/NVIDIA/nvidia-docker) is installed.

The following command builds the Docker container from the Docker file, and starts the jupyter notebook (´Demo.ipynb´):

	$ make notebook
