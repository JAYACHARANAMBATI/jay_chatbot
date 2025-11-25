
FROM python:3.11-slim


WORKDIR /app


COPY requirements.txt .


RUN apt-get update && \
    apt-get install -y git build-essential && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


COPY . .


EXPOSE 8000


ENV PYTHONUNBUFFERED=1


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
