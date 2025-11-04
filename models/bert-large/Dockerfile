FROM python:3.11-slim
WORKDIR /app
COPY server.py qa_pb2.py qa_pb2_grpc.py /app/

RUN pip install --no-cache-dir torch==2.3.0+cpu torchvision==0.18.0+cpu \
    -f https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers grpcio protobuf


ENV MODEL_NAME=bert-large-uncased-whole-word-masking-finetuned-squad
ENV PORT=50051
CMD ["python", "server.py"]
