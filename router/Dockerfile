FROM python:3.10-slim

WORKDIR /app

COPY router.py qa.proto qa_pb2.py qa_pb2_grpc.py ./

RUN pip install --no-cache-dir grpcio grpcio-tools

EXPOSE 50050

CMD ["python", "router.py"]
