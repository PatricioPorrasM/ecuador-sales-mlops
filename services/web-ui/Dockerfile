FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py kafka_producer.py metrics.py ./
COPY templates/ ./templates/

EXPOSE 5002

CMD ["gunicorn", \
     "--bind", "0.0.0.0:5002", \
     "--workers", "2", \
     "--threads", "2", \
     "--timeout", "150", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
