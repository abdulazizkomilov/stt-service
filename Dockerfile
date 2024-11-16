# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . /app

# Install FastAPI, Uvicorn, transformers, accelerate, and other necessary Python libraries
RUN pip install --no-cache-dir fastapi uvicorn transformers torch torchvision torchaudio accelerate
RUN pip install --no-cache-dir -r requirements.txt

# Install additional system dependencies for FFmpeg (required for audio processing)
RUN apt-get update && apt-get install -y ffmpeg

# Expose the default FastAPI port
EXPOSE 8000

# Run the FastAPI app with Uvicorn and multiple workers
CMD ["python", "main.py"]
