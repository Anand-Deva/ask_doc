# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    curl \
    poppler-utils \
    libmagic1 \
    ghostscript \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories with proper permissions
RUN mkdir -p /app/temp && \
    mkdir -p /app/uploads && \
    chmod 777 /app/temp && \
    chmod 777 /app/uploads

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create non-root user and switch to it
RUN useradd -m appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

USER appuser

# Expose the port Streamlit runs on
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501 \
    TEMP_UPLOAD_DIR=/app/temp

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"] 