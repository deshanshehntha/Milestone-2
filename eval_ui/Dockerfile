FROM python:3.11-slim

WORKDIR /app

# System deps (often needed by sentence-transformers / tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy UI + your eval script + grpc stubs
# Put these files into ./eval_ui/
COPY app.py /app/app.py
COPY qa_pb2.py /app/qa_pb2.py
COPY qa_pb2_grpc.py /app/qa_pb2_grpc.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
