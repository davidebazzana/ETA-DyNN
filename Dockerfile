FROM nvcr.io/nvidia/pytorch:26.02-py3-igpu

RUN mkdir /usr/src/ETA-DyNN
WORKDIR /usr/src/ETA-DyNN
COPY . .

RUN pip install -e .
RUN pip install -r ./ee_cnn/requirements.txt
RUN pip install -r ./key_frame_extraction/requirements.txt