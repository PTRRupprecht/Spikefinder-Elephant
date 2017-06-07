help:
	@cat Makefile

DATA?="${HOME}/Data"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
# DOCKER=docker
BACKEND=tensorflow
TEST=tests/
SRC=$(shell dirname `pwd`)

build:
	docker build -t keras --build-arg python_version=3.5 -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -it -v $(SRC):/elephant -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) keras bash

notebook: build
	$(DOCKER) run -it -v $(SRC):/elephant -v $(DATA):/data --net=host --env KERAS_BACKEND=$(BACKEND) keras

