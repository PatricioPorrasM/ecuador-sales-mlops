FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY consumer.py metrics.py ./

# Prometheus metrics endpoint
EXPOSE 8003

# Run directly with Python — no WSGI server needed.
# SIGTERM is handled gracefully in the consumer loop.
CMD ["python", "consumer.py"]
