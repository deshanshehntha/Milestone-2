FROM python:3.11-slim
WORKDIR /app
COPY server.py qa_pb2.py qa_pb2_grpc.py /app/

RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 && \
    pip install --no-cache-dir transformers grpcio protobuf minio

ENV MODEL_NAME=tinyroberta
ENV MODEL_ID=deepset/tinyroberta-squad2
ENV PORT=50051
CMD ["python", "server.py"]
