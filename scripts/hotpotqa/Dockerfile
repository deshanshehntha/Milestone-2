FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1
ENV TORCH_DISABLE_CUDA=1

WORKDIR /app
COPY prepare_hotpot.py ./

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    # --- install CPU-only torch first ---
    pip install --no-cache-dir torch==2.3.0+cpu torchvision==0.18.0+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    # --- base packages ---
    pip install --no-cache-dir pandas tqdm "datasets<4.0.0" && \
    # --- huggingface compatibility ---
    pip install --no-cache-dir huggingface_hub==0.14.1 && \
    # --- vector + embeddings ---
    pip install --no-cache-dir "chromadb[cpu]" sentence-transformers==2.2.2 && \
    # --- text splitters ---
    pip install --no-cache-dir langchain==0.2.14  && \
    # --- dataset source ---
    pip install --no-cache-dir kagglehub && \
    # --- cleanup ---
    apt-get remove -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

CMD ["python", "prepare_hotpot.py"]
