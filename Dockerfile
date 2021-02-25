# PyTorch GPU
FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

# Port 8888 for Jupyter lab; 6006 for Tensorboard
EXPOSE 8888 6006

# Set working directory
WORKDIR /root

# Install apt-get packages
RUN apt-get -y update \
    && apt-get -y install vim less \
    && apt-get -y install libglib2.0-0 libgl1-mesa-glx \
    && apt-get -y install libsm6 libxext6 libxrender-dev \
    && apt-get -y install ffmpeg

# Install Jupyter lab and configure
RUN pip install -U --upgrade pip \
    && pip install -U jupyterlab \
    && cd /root && jupyter lab -y --generate-config \
    && sed -i "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '0.0.0.0'/g" /root/.jupyter/jupyter_lab_config.py \
    && sed -i "s/#c.NotebookApp.open_browser = True/c.NotebookApp.open_browser = False/g" /root/.jupyter/jupyter_lab_config.py \
    && sed -i "s/#c.NotebookApp.terminado_settings = {}/c.NotebookApp.terminado_settings = {'shell_command': ['bash']}/g" /root/.jupyter/jupyter_lab_config.py

# Install pip requirements
COPY requirements.txt /root/requirements.txt
RUN pip3 install Cython && pip3 install -r /root/requirements.txt

# Install DCNv2
RUN cd /root && git clone https://github.com/CharlesShang/DCNv2 && \
  cd DCNv2 && ./make.sh

# Set bash as shell (for Jupyter lab)
ENV SHELL=/bin/bash

# Start Jupyter lab
CMD jupyter lab --allow-root
