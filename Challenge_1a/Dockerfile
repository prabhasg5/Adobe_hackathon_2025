# Use official lightweight Python base image (compatible with amd64)
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirement file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your source code
COPY . .

# Run permissions and defaults
ENV PYTHONUNBUFFERED=1

# Run the main script when the container starts
ENTRYPOINT ["python", "main.py"]







