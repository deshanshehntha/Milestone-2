FROM python:3.10-slim

WORKDIR /app
COPY prepare_hotpot.py ./

RUN pip install --no-cache-dir pandas kagglehub chromadb sentence-transformers tqdm

CMD ["python", "prepare_hotpot.py"]
