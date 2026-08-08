FROM ubuntu:22.04

LABEL maintainer="Thein Kyaw Lwin" description="Ubuntu environment customized for KenLM training and evaluation"

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Update and install essentials + KenLM dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    apt-utils \
    curl \
    wget \
    git \
    vim \
    nano \
    iputils-ping \
    unzip \
    jq \
    tree \
    software-properties-common \
    cmake \
    # Compilers and Build Tools
    build-essential \
    # Compression Libraries (KenLM uses these for compressed ARPA files)
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    # Boost and Unicode Libraries (KenLM dependencies)
    libboost-all-dev \
    libicu-dev \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Pre-install Matplotlib, Pandas, Hugging Face Datasets, and PyArrow
RUN pip3 install --no-cache-dir matplotlib pandas datasets pyarrow

# Set pip alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory to project workspace
WORKDIR /workspace

# Nice prompt
RUN echo 'export PS1="\[\033[01;32m\]kenlm@container\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> /root/.bashrc

CMD ["/bin/bash"]
