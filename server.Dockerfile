FROM docker.io/pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

RUN mkdir /usr/src/ETA-DyNN
RUN mkdir /usr/src/vit4v
WORKDIR /usr/src/ETA-DyNN