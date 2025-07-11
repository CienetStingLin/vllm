FROM google/cloud-sdk:latest

RUN apt-get update && apt-get install -y \
    wget git sudo build-essential \
    libopenblas-base libopenmpi-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN mkdir -p /root/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh && \
    bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && \
    rm -rf /root/miniconda3/miniconda.sh

ENV PATH="/root/miniconda3/bin:$PATH"

# create a conda environment for vllm
RUN conda create -n vllm python=3.11 -y

# install vllm
SHELL ["/bin/bash", "-c"]
RUN source activate vllm && \
    git clone https://github.com/vllm-project/vllm.git /root/vllm && \
    cd /root/vllm && \
    pip uninstall torch torch-xla -y && \
    pip install -r requirements/tpu.txt && \
    VLLM_TARGET_DEVICE="tpu" pip install --editable . && \
    pip install pandas datasets

# Set environment variables
WORKDIR /root/vllm


COPY entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

ENTRYPOINT ["/root/entrypoint.sh"]
