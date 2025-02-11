# Use PyTorch 2.0.1 with CUDA 11.7 as base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /opt/ml/code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up environment variables for better PyTorch performance
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CUDA_LAUNCH_BLOCKING="1"
ENV PYTHONUNBUFFERED="1"

# Copy source code
COPY src/ src/
COPY src/sagemaker_train.py .

# Set Python path
ENV PYTHONPATH=/opt/ml/code

# Create directories for data and model artifacts
RUN mkdir -p /opt/ml/model \
    /opt/ml/input/data/training \
    /opt/ml/input/data/validation \
    /opt/ml/output

# Set up entrypoint
ENTRYPOINT ["python", "sagemaker_train.py"]

# Add health check
HEALTHCHECK --interval=5m --timeout=3s \
  CMD curl -f http://localhost:8080/ping || exit 1

# Label the image
LABEL maintainer="EZ Ridge Team" \
      version="1.0" \
      description="Training container for EZ Ridge roof analysis model"