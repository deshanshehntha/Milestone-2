FROM python:3.10-slim

WORKDIR /app

# Copy router + model router + S3 helper + protos
COPY router.py model_router.py s3_artifacts.py qa.proto qa_pb2.py qa_pb2_grpc.py ./

RUN pip install --no-cache-dir grpcio grpcio-tools joblib pandas scikit-learn boto3

EXPOSE 50050
CMD ["python", "router.py"]
