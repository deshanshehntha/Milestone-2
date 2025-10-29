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

# # --- Create ChromaDB collection ---
# print(" Building ChromaDB collection for HotpotQA...")
# chroma_client = chromadb.PersistentClient(path="/mnt/ChromaDb")
# collection_name = "hpqa_data_collection"
s
# try:
#     chroma_client.delete_collection(collection_name)
#     print(f"Old collection '{collection_name}' deleted.")
# except Exception:
#     pass

# embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
# collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_fn)

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
