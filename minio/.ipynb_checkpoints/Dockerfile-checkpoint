FROM python:3.10-slim
WORKDIR /app
COPY setup_minio.py .
RUN pip install --no-cache-dir minio torch transformers
CMD ["python", "setup_minio.py"]