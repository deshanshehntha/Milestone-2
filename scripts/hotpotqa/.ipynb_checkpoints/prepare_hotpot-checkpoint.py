import os
import json
import time
import pandas as pd
import kagglehub
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SentenceTransformersTokenTextSplitter
from tqdm import tqdm

# --- Setup paths ---
os.makedirs("/mnt/HotpotQA", exist_ok=True)
os.makedirs("/mnt/ChromaDb", exist_ok=True)

print("📥 Downloading HotpotQA dataset via KaggleHub...")
path = kagglehub.dataset_download("jeromeblanchet/hotpotqa-question-answering-dataset")
print(f"✅ Path to HotpotQA dataset: {path}")

json_path = os.path.join(path, "hotpot_dev_distractor_v1.json")
if not os.path.exists(json_path):
    raise FileNotFoundError(f"Could not find {json_path}")

with open(json_path, "r") as f:
    hotpot_data = json.load(f)

print(f"✅ Loaded {len(hotpot_data)} examples")

rows = []
for ex in hotpot_data:
    rows.append({
        "id": ex["_id"],
        "question": ex["question"],
        "answer": ex["answer"],
        "context": ex["context"],
        "supporting_facts": ex["supporting_facts"]
    })

df = pd.DataFrame(rows)
out_csv = "/mnt/HotpotQA/processed_hotpot_df.csv"
df.to_csv(out_csv, index=False)
print(f"✅ Saved processed HotpotQA CSV to {out_csv}")

# --- Config ---
TOKEN_CHUNK_SIZE = 256
TOKEN_CHUNK_OVERLAP = 10
CHAR_CHUNK_SIZE = 1000
CHAR_CHUNK_OVERLAP = 10


# --- Connect to Chroma with retry logic ---
for attempt in range(10):
    try:
        chroma_client = chromadb.HttpClient(
            host="chroma",
            port=8000,
            settings=Settings(allow_reset=True)
        )
        print("✅ Connected to Chroma server.")
        break
    except Exception as e:
        print(f"⏳ Waiting for Chroma to be ready (attempt {attempt+1}/10)...")
        time.sleep(5)
else:
    raise ConnectionError("❌ Could not connect to Chroma server after 10 attempts.")

embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(
    name="hotpotqa_data_collection",
    embedding_function=embedding_fn
)

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=CHAR_CHUNK_SIZE,
    chunk_overlap=CHAR_CHUNK_OVERLAP
)

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=TOKEN_CHUNK_OVERLAP,
    tokens_per_chunk=TOKEN_CHUNK_SIZE
)


def ingest_hotpot_to_chroma(hotpot_df, chroma_collection, character_splitter, token_splitter):
    for q_id, group in tqdm(hotpot_df.groupby("id"), desc="📚 Processing HotpotQA"):
        context_blocks = []
        for title, paragraphs in group.iloc[0]["context"]:
            section_text = "\n".join(paragraphs)
            context_blocks.append(f"{title}\n{section_text}")
        full_text = "\n\n".join(context_blocks)

        char_chunks = character_splitter.split_text(full_text)
        token_chunks = []
        for chunk in char_chunks:
            token_chunks.extend(token_splitter.split_text(chunk))

        if not token_chunks:
            print(f"⚠️ Skipping {q_id}: no chunks generated.")
            continue

        ids = [f"{q_id}_{i}" for i in range(len(token_chunks))]
        metadatas = [{"hotpot_id": q_id} for _ in token_chunks]

        chroma_collection.add(
            documents=token_chunks,
            ids=ids,
            metadatas=metadatas
        )

    print("✅ All HotpotQA documents embedded and stored in Chroma.")


if __name__ == "__main__":
    ingest_hotpot_to_chroma(df, collection, character_splitter, token_splitter)
