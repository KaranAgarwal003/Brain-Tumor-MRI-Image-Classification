# Use Python 3.10.8 as base image
FROM python:3.10.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for Pillow, TensorFlow, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (default for Flask)
EXPOSE 5000

# Start the app (adjust this line if your entry point is not app.py)
CMD ["python", "app.py"]
