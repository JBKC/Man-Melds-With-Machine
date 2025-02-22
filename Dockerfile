# Base image with Python 3.11
FROM python:3.11-slim

# Install system dependencies, including FFmpeg
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    ffmpeg && \
    apt-get clean

# Set working directory inside the container
WORKDIR /app

# Copy the local project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir mediapipe opencv-python

# Expose the RTSP port
EXPOSE 8554

# Default command to start the RTSP processing
CMD ["bash", "-c", "ffmpeg -i rtsp://host.docker.internal:8554/live -f rawvideo -pix_fmt yuv420p pipe:1 | python main_script.py"]

