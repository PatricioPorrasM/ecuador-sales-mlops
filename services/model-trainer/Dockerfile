FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código fuente del entrenador.
# Los datos se proveen en tiempo de ejecución:
#   - docker-compose : volume mount  ./data:/app/data:ro
#   - Kubernetes     : PersistentVolumeClaim montado en /app/data
COPY trainer.py train_v1.py train_v2.py compare_and_promote.py ./

# Volumen de salida compartido con ml-inference via PersistentVolumeClaim
VOLUME ["/app/models"]

# Default entry point; K8s Jobs override this per-phase
CMD ["python", "train_v1.py"]
