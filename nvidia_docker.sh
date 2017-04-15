#!/bin/bash
nvidia-docker run -it -p 4200:8888 -v $(pwd):/notebooks gcr.io/tensorflow/tensorflow:latest-gpu-py3 bash
