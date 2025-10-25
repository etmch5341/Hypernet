# Use Python 3.12.6 as base image
FROM python:3.12.6-slim-bookworm

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Update and install system dependencies (includes GDAL/proj needed by fiona)
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
    gdal-bin \
    libgdal-dev \
    proj-bin \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pyosmium (Python bindings for osmium)
# RUN pip3 install --no-cache-dir pyosmium

# Verify Python version is 3.12.6
RUN python3 --version && \
    python3 -c "import sys; assert sys.version_info[:3] == (3, 12, 6), f'Expected Python 3.12.6, got {sys.version}'"

# Copy requirements.txt into the image (if you have one)
# Note: Docker COPY requires the file to exist in the build context. If you
# don't have a requirements.txt, either add one or remove this line.
COPY requirements.txt /workspace/

# Install requirements if the file exists
RUN if [ -f /workspace/requirements.txt ]; then \
        pip3 install --no-cache-dir -r /workspace/requirements.txt; \
    fi

# Install esy-osmfilter from PyPI (original version)
RUN pip3 install --no-cache-dir esy-osmfilter

# Copy only your modified files into the container
# These will overwrite the original versions
COPY ./modified_files/element_filter.py /usr/local/lib/python3.12/site-packages/esy/osmfilter/element_filter.py
COPY ./modified_files/osm_filter.py /usr/local/lib/python3.12/site-packages/esy/osmfilter/osm_filter.py
COPY ./modified_files/pre_filter.py /usr/local/lib/python3.12/site-packages/esy/osmfilter/pre_filter.py

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