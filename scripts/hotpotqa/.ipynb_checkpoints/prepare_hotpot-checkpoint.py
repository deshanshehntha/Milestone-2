import os
import json
import pandas as pd
import kagglehub
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

# --- Setup paths ---
os.makedirs("/mnt/HotpotQA", exist_ok=True)
os.makedirs("/mnt/ChromaDb", exist_ok=True)

print("Downloading HotpotQA dataset via KaggleHub...")
path = kagglehub.dataset_download("jeromeblanchet/hotpotqa-question-answering-dataset")
print(f"Path to HotpotQA dataset: {path}")

json_path = os.path.join(path, "hotpot_dev_distractor_v1.json")
if not os.path.exists(json_path):
    raise FileNotFoundError(f"Could not find {json_path}")

with open(json_path, "r") as f:
    hotpot_data = json.load(f)

print(f"Loaded {len(hotpot_data)} examples")

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
print(f"Saved processed HotpotQA CSV to {out_csv}")


DATASET_HOTPOT = "HotpotQA"

TOKEN_CHUNK_SIZE = 256
TOKEN_CHUNK_OVERLAP = 10

CHAR_CHUNK_SIZE = 1000
CHAR_CHUNK_OVERLAP = 10


# Connect to the Chroma server
chroma_client = chromadb.HttpClient(
    host="chroma",
    port=8000,
    settings=Settings(allow_reset=True)
)

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

    for q_id, group in tqdm(hotpot_df.groupby("id"), desc="Processing Hotpot QA"):

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
            print(f"Skipping question {q_id}: no context found produced.")
            continue

        ids = [f"{q_id}_{i}" for i in range(len(token_chunks))]
        
        metadatas = [
            {
                "hotpot_id": q_id
            }
            for _ in token_chunks
        ]

        chroma_collection.add(
            documents=token_chunks,
            ids=ids,
            metadatas=metadatas
        )

    print("All HotpotQA questions processed and stored in Chroma.")

ingest_hotpot_to_chroma(df, collection, character_splitter, token_splitter)

# for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding HotpotQA"):
#     context_blocks = []
#     for title, paragraphs in row["context"]:
#         section = f"{title}\n" + "\n".join(paragraphs)
#         context_blocks.append(section)
#     full_text = "\n\n".join(context_blocks)

#     chunks = [full_text[i:i + 1000] for i in range(0, len(full_text), 1000)]
#     ids = [f"{row['id']}_{i}" for i in range(len(chunks))]
#     metadatas = [{"hotpot_id": row["id"]} for _ in chunks]
#     collection.add(documents=chunks, ids=ids, metadatas=metadatas)

# print(f" ChromaDB HotpotQA collection '{collection_name}' ready at /mnt/ChromaDb")
