FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

ARG PYTHON_VERSION=3.7

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   cmake \
                   git \
                   curl \
                   ca-certificates \
                   && \
    rm -rf /var/lib/apt/lists

# add non-root user
RUN useradd --create-home --shell /bin/bash containeruser
USER containeruser
WORKDIR /home/containeruser
COPY ./ ./transition/

# install conda and python
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /home/containeruser/conda && \
    rm ~/miniconda.sh && \
    /home/containeruser/conda/bin/conda clean -ya && \
    /home/containeruser/conda/bin/conda install -y python=$PYTHON_VERSION

# add conda to path
ENV PATH /home/containeruser/conda/bin:$PATH

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r ./transition/requirements.txt

ENV CHECKPOINT "ytlin/verr5re0"

WORKDIR /home/containeruser/transition/
CMD python3 demo.py ${CHECKPOINT}