FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1
ENV TORCH_DISABLE_CUDA=1

WORKDIR /app

# Copy the script (replace below with the correct filename per service)
COPY prepare_qasper.py ./
# or COPY prepare_qasper.py ./

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir torch==2.3.0+cpu torchvision==0.18.0+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir pandas tqdm "datasets<4.0.0" && \
    # --- pin compatible Hugging Face versions ---
    pip install --no-cache-dir huggingface-hub==0.14.1 sentence-transformers==2.2.2 && \
    # --- install Chroma and LangChain ---
    pip install --no-cache-dir "chromadb[cpu]" langchain==0.2.14  && \
    # --- dataset downloader ---
    pip install --no-cache-dir kagglehub && \
    # --- clean up ---
    apt-get remove -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Run script
CMD ["python", "prepare_qasper.py"]
