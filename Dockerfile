FROM docker.deepsystems.io/supervisely/five/base-pytorch-1-3-cuda-10-1:master

RUN pip install yacs
RUN pip install cython
RUN pip install cython-bbox
RUN pip install numba
RUN pip install progress
RUN pip install motmetrics
RUN pip install lap
RUN pip install openpyxl
RUN pip install tensorboardX
RUN conda install torchvision==0.4.0 cudatoolkit=10.0
RUN pip install supervisely

COPY commands.sh /commands.sh
RUN ["chmod", "+x", "/commands.sh"]
ENTRYPOINT ["sh", "/commands.sh"]