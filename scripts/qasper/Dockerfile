FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1
ENV TORCH_DISABLE_CUDA=1

WORKDIR /app
COPY prepare_qasper.py ./

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir torch==2.3.0+cpu torchvision==0.18.0+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    # --- Pin datasets first (compatible with huggingface-hub <0.15)
    pip install --no-cache-dir datasets==2.14.6 && \
    # --- Huggingface + transformers stack
    pip install --no-cache-dir huggingface-hub==0.14.1 sentence-transformers==2.2.2 && \
    # --- Core tools
    pip install --no-cache-dir pandas tqdm && \
    # --- Vector DB + splitter
    pip install --no-cache-dir "chromadb[cpu]" langchain==0.2.14 && \
    # --- Dataset loader
    pip install --no-cache-dir kagglehub && \
    # --- Cleanup
    apt-get remove -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

CMD ["python", "prepare_qasper.py"]
