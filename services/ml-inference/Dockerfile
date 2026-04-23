FROM python:3.11-slim

WORKDIR /app

# Install dependencies before copying source for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py model_loader.py kafka_producer.py metrics.py ./

# Shared PersistentVolumeClaim populated by the training job
# The service starts without a model and becomes ready once the file appears.
VOLUME ["/app/models"]

EXPOSE 5000

# Single worker: keeps the in-memory model in one process.
# Four threads handle concurrent requests without model duplication.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "60", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
