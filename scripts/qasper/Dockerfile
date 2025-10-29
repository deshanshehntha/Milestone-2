FROM python:3.10-slim

WORKDIR /app
COPY prepare_qasper.py ./

RUN pip install --no-cache-dir pandas chromadb sentence-transformers tqdm \
    && pip install --no-cache-dir "datasets<4.0.0"

CMD ["python", "prepare_qasper.py"]
