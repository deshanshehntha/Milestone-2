FROM python:3.10-slim
WORKDIR /app
COPY setup_minio.py .
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu torchvision==0.18.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir transformers==4.37.2 minio
CMD ["python", "setup_minio.py"]