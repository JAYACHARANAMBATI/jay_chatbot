FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required by some ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Ensure chroma persistent directory exists and is writable (mount a volume in production)
RUN mkdir -p /app/chroma_store && chmod -R 777 /app/chroma_store

# Create non-root user and adjust ownership
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8000
EXPOSE 8000

# Start the FastAPI app with Uvicorn
CMD ["uvicorn", "chatbot:app", "--host", "0.0.0.0", "--port", "8000"]
