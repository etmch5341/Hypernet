# OSM Docker Environment

This Docker container provides a complete environment for working with OpenStreetMap (OSM) data, including custom OSM library modifications and osmium tools.

## Prerequisites

- Docker installed on your machine ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose (usually comes with Docker Desktop)

## Directory Structure

```
docker-envr/
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements.txt      # Python dependencies (optional)
├── modified_files/       # Your 3 modified Python files
│   ├── element_filter.py
│   ├── osm_filter.py
│   └── pre_filter.py
├── scripts/              # Your Python scripts go here
│   ├── my_script.py
│   └── ...
├── data/                 # Place your OSM data files here
│   ├── input/            # Input data files
│   └── output/           # Output result files
└── outputs/              # Container outputs will be saved here
```

## Setup Instructions

### 1. Build the Docker Image

```bash
docker-compose build
```

This will create a Docker image with:
- Ubuntu 24.04 base
- **Python 3.12.6** (exact version)
- Osmium tools and libraries
- esy-osmfilter from PyPI with **your 3 modified files** replacing the originals
- Packages from requirements.txt (if provided)
- Additional useful packages (numpy, pandas, matplotlib, jupyter)

You will only need to rebuild if you modify the python packages or you change the docker setup.

### 2. Run the Container

#### Option A: Interactive Shell

```bash
docker-compose run --rm osm-environment
```

This starts an interactive bash shell inside the container.

#### Option B: Run as a Service (recommended)
NOTE: Won't disapear when you close terminal

```bash
docker-compose up -d
```

Then connect to it:

```bash
docker exec -it osm-workspace bash
```

### 3. Stop the Container

```bash
docker-compose down
```

## Using the Container

### Verifying Python Version

To confirm you're using Python 3.12.6:

```bash
# Method 1: Quick check
docker-compose run --rm osm-environment python3 --version

# Method 2: Detailed verification
docker-compose run --rm osm-environment python3 /workspace/verify_python.py

# Method 3: From inside the container
docker-compose run --rm osm-environment
python3 --version
```

Expected output: `Python 3.12.6`

### Running Your Python Files

There are several ways to run your Python scripts in the container:

#### Method 1: Run from Inside the Container (Interactive)

1. Start an interactive shell:
```bash
docker-compose run --rm osm-environment
```

2. Inside the container, run your script:
```bash
python3 /workspace/scripts/my_script.py
# or navigate to the directory first
cd /workspace/scripts
python3 my_script.py
```

#### Method 2: Run Without Entering the Container

Execute a script directly without opening an interactive shell:

```bash
docker-compose run --rm osm-environment python3 /workspace/scripts/my_script.py
```

With arguments:
```bash
docker-compose run --rm osm-environment python3 /workspace/scripts/my_script.py arg1 arg2
```

#### Method 3: Run in a Running Container

If the container is already running:

```bash
# Start the container in the background
docker-compose up -d

# Execute your script
docker exec osm-workspace python3 /workspace/scripts/my_script.py

# OR open a shell
docker exec -it osm-workspace bash

# Then cd into the scripts directory and run script.py
cd scripts
python3 ./my_script.py
```

## Modifying or Adding Files
Mounting creates a bridge between your computer (host) and the Docker container. Think of it like a window between two rooms.

Since the `./data`, `./outputs`, and `./scripts` are mounted (look at docker-compose.yml):

1. Files appear in both places - When you put a file in `./data/` on your computer, it's instantly visible at `/workspace/data/` inside the container

2. Two-way sync - Changes go both ways:
- Put a .osm.pbf file in `./data/input/` → Container sees it
- Script writes output to `/workspace/data/output/` → You see it in `./data/output/`

3. No copying needed - Without mounting, you'd have to copy files into the container. With mounting, they're shared automatically.

### Using Osmium Tools

The container includes osmium command-line tools:

```bash
osmium --help
osmium fileinfo /workspace/data/your-file.osm.pbf
```

## Working with Files

### Adding OSM Data Files

Place your `.osm`, `.osm.pbf`, or `.osm.bz2` files in the `data/` directory on your host machine. They will be accessible at `/workspace/data/` inside the container.

### Saving Outputs

Any files you save to `/workspace/outputs/` inside the container will be accessible in the `outputs/` directory on your host machine.

## Updating Your Modified Files

If you make changes to your modified Python files on your host machine:

1. The changes are in the `modified_files/` directory
2. Rebuild the container to apply the changes:

```bash
docker-compose down
docker-compose build
docker-compose up -d
```

**Quick rebuild:**
```bash
docker-compose build --no-cache
```

## Customization

### Adding More Python Packages

**Method 1: Using requirements.txt (Recommended)**

Add packages to your `requirements.txt` file:

```txt
# requirements.txt
geopandas>=0.14.0
shapely>=2.0.0
folium>=0.14.0
```

Then rebuild:

```bash
docker-compose build
```

**Method 2: Direct Installation in Running Container (Temporary)**

Install packages in a running container (won't persist after rebuild):

```bash
docker exec osm-workspace pip3 install package-name
```

**Method 3: Edit Dockerfile (For Permanent Base Packages)**

Edit the Dockerfile and add packages to the pip install command:

```dockerfile
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    your-package-here
```

Then rebuild:

```bash
docker-compose build
```