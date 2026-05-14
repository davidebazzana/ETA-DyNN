FROM nvcr.io/nvidia/pytorch:26.02-py3-igpu

RUN mkdir /usr/src/ETA-DyNN
WORKDIR /usr/src/ETA-DyNN