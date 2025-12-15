FROM python:3.10-slim

WORKDIR /app
COPY setup_minio.py .

# Install CPU torch (keep your existing approach)
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu torchvision==0.18.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# ✅ Upgrade HF stack for Phi-3.5 remote code support
RUN pip install --no-cache-dir -U \
    "transformers>=4.45.0" \
    "huggingface_hub>=0.24.0" \
    "accelerate>=0.33.0" \
    "tokenizers>=0.19.1" \
    "safetensors>=0.4.3" \
    minio

CMD ["python", "setup_minio.py"]
