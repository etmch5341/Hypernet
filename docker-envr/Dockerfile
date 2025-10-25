# Use Python 3.12.6 as base image
FROM python:3.12.6-slim-bookworm

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libbz2-dev \
    libexpat1-dev \
    zlib1g-dev \
    libboost-dev \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-thread-dev \
    libboost-iostreams-dev \
    libosmium2-dev \
    osmium-tool \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pyosmium (Python bindings for osmium)
RUN pip3 install --no-cache-dir pyosmium

# Verify Python version is 3.12.6
RUN python3 --version && \
    python3 -c "import sys; assert sys.version_info[:3] == (3, 12, 6), f'Expected Python 3.12.6, got {sys.version}'"

# Copy your modified esy-osmfilter library into the container
# This assumes you have a local 'esy-osmfilter' directory with your modifications
COPY ./esy-osmfilter /workspace/esy-osmfilter

# Install your modified esy-osmfilter library in editable mode
RUN cd /workspace/esy-osmfilter && pip3 install -e .

# Install additional Python packages that might be useful
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    jupyter

# Set Python path to include workspace
ENV PYTHONPATH="${PYTHONPATH}:/workspace"

# Create a non-root user for better security
RUN useradd -m -s /bin/bash osmuser && \
    chown -R osmuser:osmuser /workspace

USER osmuser

# Set the default command
CMD ["/bin/bash"]