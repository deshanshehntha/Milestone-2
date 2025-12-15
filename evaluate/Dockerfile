FROM python:3.10-slim

WORKDIR /app

# Copy evaluation script and any required files
COPY evaluate/evaluate_pipeline.py ./
COPY . .

# Install dependencies
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu torchvision==0.18.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir \
    grpcio grpcio-tools pandas numpy sentence-transformers transformers tqdm chromadb datasets

# Create results directory
RUN mkdir -p /mnt/results

CMD ["python", "evaluate_pipeline.py"]
