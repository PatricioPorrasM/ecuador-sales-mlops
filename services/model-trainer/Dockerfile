FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source files — data is baked in for portability; override DATA_PATH env
# var or mount a PVC at /app/data in production to use a different dataset.
COPY trainer.py train_v1.py train_v2.py compare_and_promote.py ./
COPY data/ ./data/

# Output volume shared with ml-inference via PersistentVolumeClaim
VOLUME ["/app/models"]

# Default entry point; K8s Jobs override this per-phase
CMD ["python", "train_v1.py"]
