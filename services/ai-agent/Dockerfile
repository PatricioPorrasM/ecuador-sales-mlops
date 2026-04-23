FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py agent.py tools.py kafka_producer.py metrics.py ./

# Datos SRI — montados desde el directorio data/ en la raíz del repo.
# docker-compose: volumes: ["./data:/app/data:ro"]
# Kubernetes    : PVC de solo lectura montado en /app/data
# DATA_PATH     : sobreescribe la ruta predeterminada /app/data/Bdd_SRI_2025.csv
VOLUME ["/app/data"]

EXPOSE 5001

# Single worker keeps the CSV cache in one process.
# Longer timeout (120 s) allows for multi-step LLM + tool latency.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5001", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
