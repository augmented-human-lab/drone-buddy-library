# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements_docker.txt first to leverage Docker cache
COPY requirements_docker.txt /app/requirements.txt

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install setuptools-rust
RUN pip install setuptools-rust==1.7.0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-dev \
    portaudio19-dev \
    rustc \
    cargo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install specific Python dependencies first if not included in requirements_docker.txt
# If these are already in requirements_docker.txt, this step might be redundant

# Install the rest of the Python dependencies
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy the rest of your library's source code and tests into the container
COPY . /app

# Specify the default command to run the test script
CMD ["python", "./test/test_docker.py"]
