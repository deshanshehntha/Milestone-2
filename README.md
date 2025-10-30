# Milestone 2: Containerized Question-Answering Pipeline

## Overview
This milestone focuses on building and evaluating a **containerized Question Answering (QA) pipeline** composed of multiple model variants deployed via **Docker Compose**.  
The primary goal is to analyze how different model configurations affect **accuracy, latency, throughput, and resource consumption**.

Our setup demonstrates a multi-model orchestration pipeline that includes:
- Data preparation for **HotpotQA** and **QASPER** datasets  
- Multiple **QA model variants** (TinyRoBERTa, RoBERTa-Base, BERT-Large)  
- A **router service** managing requests between models using **gRPC**  
- An **evaluation container** to analyze accuracy, latency, and scaling metrics  

---

### Key Components

| Service | Description |
|----------|--------------|
| **prepare-hotpotqa** | Prepares the HotpotQA dataset and stores it in `/mnt/HotpotQA` |
| **prepare-qasper** | Prepares the QASPER dataset and stores it in `/mnt/Qasper` |
| **tinyroberta / roberta-base / bert-large** | Individual QA models containerized for scalable deployment |
| **qa-router** | A gRPC-based router that distributes requests to model backends |
| **evaluate-pipeline** | Evaluation module that measures latency, cost, and accuracy for each configuration |

---

---

## Current Progress

### Completed
1. **Created initial containerized model variants** for TinyRoBERTa, RoBERTa-Base, and BERT-Large.  
2. **Integrated gRPC communication** between router and model containers.  
3. **Prepared datasets (HotpotQA and QASPER)** using dedicated data preparation containers.  
4. **Set up shared ChromaDB** for consistent embedding and retrieval storage.  
5. **Connected evaluation container** to measure pipeline latency, throughput, and accuracy.

---

## Next Steps (Ongoing Development)

### Performance Analysis
- Measure **per-step latency** and identify the most time-consuming components.  
- Create **accuracy vs. latency** and **accuracy vs. throughput** plots.  
- Compare scaling effects by testing with different replicas and CPU allocations.

### Optimization Tasks
- Apply **TensorRT** and **ONNX graph optimizations** for model variants.  
- Explore **quantization levels** to reduce model inference time.  
- Conduct **horizontal scaling experiments** (replicas × CPU cores) and compute total cost.  
- Determine **SLA compliance** (e.g., 500 ms latency, 20 req/s).

### Build and start all containers
In the project root (Milestone-2) (where your docker-compose.yml file is located), run:

docker compose up --build

---

## Conclusion
This milestone successfully establishes a modular, containerized, and scalable QA pipeline integrating multiple model backends via gRPC.  
The next phase will focus on **performance benchmarking**, **optimization**, and **identifying the best trade-off between accuracy, latency, and cost** to meet defined SLAs.  
The project is currently an **ongoing development**, with optimization and evaluation experiments in progress.
