FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1
ENV TORCH_DISABLE_CUDA=1

WORKDIR /app
COPY prepare_qasper.py ./

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir torch==2.3.0+cpu torchvision==0.18.0+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir pandas tqdm "datasets<4.0.0" && \
    pip install --no-cache-dir "chromadb[cpu]" sentence-transformers==2.2.2 && \
    pip install --no-cache-dir langchain==0.2.14 langchain-experimental==0.0.67 && \
    pip install --no-cache-dir kagglehub && \
    apt-get remove -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

CMD ["python", "prepare_qasper.py"]
