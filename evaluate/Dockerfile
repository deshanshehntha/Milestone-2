FROM python:3.10-slim

WORKDIR /app

COPY evaluate/evaluate_pipeline.py ./
COPY . .

# Install all dependencies
RUN pip install --no-cache-dir \
    grpcio \
    grpcio-tools \
    pandas \
    numpy \
    sentence-transformers \
    transformers \
    tqdm

# Create results directory
RUN mkdir -p /mnt/results

CMD ["python", "evaluate_pipeline.py"]
