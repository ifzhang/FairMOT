#########################################################
FROM nvidia/cuda:10.2-devel-ubuntu18.04 AS builder

## Use bash to support string substitution.
#SHELL ["/bin/bash", "-c"]

# System dependencies setup, install them as root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt-get update
RUN apt-get install -y apt-utils curl wget tzdata software-properties-common
RUN apt-get install -y ffmpeg
RUN apt-get install -y libgtk2.0-0 libcanberra-gtk-module
#RUN apt-get install -y nvidia-cuda-toolkit

# System dependencies for OpenCV
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# Additional system dependencies, also for python and scientific python
ARG PYTHON_VERSION=3.8
ARG PYTHON_MAJOR_VERSION=3
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y build-essential libssl-dev git ca-certificates libjpeg-dev libpng-dev make cmake
RUN apt-get install -y python3-distutils python3-apt python${PYTHON_VERSION}-tk python${PYTHON_VERSION} python${PYTHON_VERSION}-dev

# Get pip so that it uses specific python version and symlinks to set defaults
RUN curl https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN ln -s /usr/bin/pip${PYTHON_MAJOR_VERSION} /usr/bin/pip

# Install Miniconda package manger.
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Install conda packages.
ENV PATH="/opt/conda/bin:$PATH"
RUN conda install python=${PYTHON_VERSION}
RUN conda update -qy conda
RUN conda install -y -c pytorch pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2
COPY requirements.txt /app/
RUN pip install cython
RUN pip install numpy
RUN pip install -r /app/requirements.txt
RUN rm /app/requirements.txt

# Environment variables for python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=TRUE \
    PYTHONPATH=/app/FairMOT

WORKDIR /app

###################################################
FROM gcr.io/google.com/cloudsdktool/cloud-sdk:slim as gcloud

#########################################################
FROM builder AS local

# Install latest gcloud sdk
COPY --from=gcloud /usr/lib/google-cloud-sdk /usr/lib/google-cloud-sdk
ENV PATH /usr/lib/google-cloud-sdk/bin:$PATH

ENV HOME=/root

# Add permission to HOME & conda for host user
RUN chmod 777 -R $HOME
RUN chmod 777 -R /opt/conda

WORKDIR /app

RUN ["/bin/bash"]
