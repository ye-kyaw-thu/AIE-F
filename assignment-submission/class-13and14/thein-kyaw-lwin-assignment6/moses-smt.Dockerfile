FROM ubuntu:22.04

LABEL maintainer="Thein Kyaw Lwin" description="Ubuntu environment customized for Moses SMT and g2p/p2g tasks"

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Update and install essentials + Moses dependencies
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
    # Compilers and Build Tools (required for runtime linking/extensions)
    build-essential \
    # Compression Libraries (Moses uses these for phrase tables)
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    # Boost and Unicode Libraries (Moses and MGIZA dependency)
    libboost-all-dev \
    libicu-dev \
    # Perl and required modules (For Moses training/tuning scripts)
    perl \
    libxml-twig-perl \
    libsort-naturally-perl \
    # Python 3
    python3 \
    python3-pip \
    python3-venv \
    # Graphviz (For rendering EMS workflow diagrams)
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22 LTS via NodeSource (includes npm and npx)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install fastfetch (via PPA — required for Ubuntu 22.04)
RUN add-apt-repository ppa:zhangsongcui3371/fastfetch -y && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y fastfetch && \
    rm -rf /var/lib/apt/lists/*

# Set pip alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory to project workspace
WORKDIR /workspace

# Nice prompt (optional — looks good on camera)
RUN echo 'export PS1="\[\033[01;32m\]demo@container\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> /root/.bashrc

CMD ["/bin/bash"]
