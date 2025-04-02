# Base image: PyTorch GPU image
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set the working directory:
WORKDIR /app

# Install system dependencies:
RUN apt-get update && apt-get install -y \
  python3                                \
  python3-pip                            \
  python3-dev                            \
  build-essential                        \
  libsm6                                 \
  libxext6                               \
  libxrender-dev                         \
  libglib2.0-0                           \
  && rm -rf /var/lib/apt/lists/*

# Copy the required files to the container:
COPY requirements.txt /app/

# Install Python dependencies:
RUN pip3 install --upgrade pip && \
  pip3 install --no-cache-dir -r requirements.txt

# Set environment variables to optimize GPU usage
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set the entrypoint command:
ENTRYPOINT ["python3", "unet3d.py"]
