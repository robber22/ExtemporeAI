# Use a lightweight Python base image
FROM python:3.10-slim

# Install FFmpeg (strictly required for Whisper audio processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install the required Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files
COPY app.py .

# Configure Gradio to allow external connections
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Expose the port Gradio runs on
EXPOSE 7860

# Start the application
CMD ["python", "app.py"]