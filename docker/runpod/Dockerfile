# Stage 1: uv
FROM ghcr.io/astral-sh/uv:latest AS uv

# Stage 2: oh-my-zsh
FROM ohmyzsh/zsh AS ohmyzsh

# Stage 3: Main
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

# Install uv from named stage
COPY --from=uv /uv /uvx /bin/

# Install oh-my-zsh from named stage
COPY --from=ohmyzsh /root/.oh-my-zsh /root/.oh-my-zsh
COPY --from=ohmyzsh /root/.zshrc /root/.zshrc

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openssh-server \
    ca-certificates \
    libsqlite3-0 \
    libgssapi-krb5-2 \
    libssl3 \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    libonig-dev \
    zsh \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.13 /usr/bin/python \
    && ln -sf /usr/bin/python3.13 /usr/bin/python3

# Configure SSH
RUN mkdir /var/run/sshd \
    && echo 'root:password' | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# Install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Setup zsh plugins
RUN git clone https://github.com/zsh-users/zsh-autosuggestions.git \
    /root/.oh-my-zsh/custom/plugins/zsh-autosuggestions \
    && chsh -s $(which zsh) root

# Environment variables
ENV SHELL=/bin/zsh
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set up workspace
WORKDIR /workspace

# Copy startup script
COPY --chmod=0755 docker/runpod/startup.sh /usr/local/bin/startup.sh
RUN sed -i 's/\r$//' /usr/local/bin/startup.sh

ENTRYPOINT ["/bin/bash", "-lc", "/usr/local/bin/startup.sh"]
