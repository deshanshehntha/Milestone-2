FROM python:3.11-slim

WORKDIR /app
COPY server.py qa_pb2.py qa_pb2_grpc.py /app/

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir grpcio protobuf

ENV PORT=50051
ENV OLLAMA_HOST=http://ollama:11434
ENV OLLAMA_MODEL=llama3
ENV MODEL_NAME=ollama-model

CMD ["python", "server.py"]
