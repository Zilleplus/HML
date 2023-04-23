# From the nvidea docs, this combination of tensorflow<->cuda should work:
# Tensorflow 2.11.1
# cuDNN=8.1
# CUDA=11.2

FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04
COPY requirements.txt /opt/app/requirements.txt

WORKDIR /opt/app
RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip3 install -r requirements.txt

# add the python3-tk backed, otherwise matplotlib will use agg and not show.
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-tk

# Make sure the x11 display works.
RUN echo "export DISPLAY=:0" >> ~/.bashrc

# future improvement: add TensorRT