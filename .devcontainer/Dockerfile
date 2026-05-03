FROM python:3.10-slim

# Install basic tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Default shell
CMD ["bash"]