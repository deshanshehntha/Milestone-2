FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1
ENV TORCH_DISABLE_CUDA=1
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu   # Force CPU-only builds

WORKDIR /app
COPY prepare_qasper.py ./

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    # --- install compatible PyArrow first ---
    pip install --no-cache-dir pyarrow==14.0.2 && \
    # --- Hugging Face stack ---
    pip install --no-cache-dir datasets==2.14.6 huggingface-hub==0.14.1 sentence-transformers==2.2.2 && \
    # --- torch & vector stack (CPU only, lightweight) ---
    pip install --no-cache-dir torch==2.3.0+cpu torchvision==0.18.0+cpu && \
    # --- core packages ---
    pip install --no-cache-dir pandas tqdm && \
    # --- chroma + langchain ---
    pip install --no-cache-dir "chromadb[cpu]" langchain==0.2.14 && \
    # --- dataset source ---
    pip install --no-cache-dir kagglehub && \
    # --- cleanup ---
    apt-get remove -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

CMD ["python", "prepare_qasper.py"]
