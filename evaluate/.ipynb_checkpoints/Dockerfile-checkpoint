FROM python:3.10-slim

WORKDIR /app

COPY evaluate/evaluate_pipeline.py ./
COPY . .

    
# Install lightweight CPU-only torch first
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu torchvision==0.18.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir \
    grpcio grpcio-tools pandas numpy sentence-transformers transformers tqdm


# Create results directory
RUN mkdir -p /mnt/results

CMD ["python", "evaluate_pipeline.py"]
