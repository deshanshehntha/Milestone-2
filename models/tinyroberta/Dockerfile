FROM python:3.11-slim
WORKDIR /app
COPY server.py qa_pb2.py qa_pb2_grpc.py /app/
RUN pip install --no-cache-dir torch transformers grpcio protobuf
ENV MODEL_NAME=deepset/tinyroberta-squad2
ENV PORT=50051
CMD ["python", "server.py"]
