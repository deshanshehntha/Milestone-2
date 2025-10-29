import os
import pandas as pd
from datasets import load_dataset
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

os.makedirs("/mnt/Qasper", exist_ok=True)
os.makedirs("/mnt/ChromaDb", exist_ok=True)

print("Loading QASPER dataset from HuggingFace...")
qasper_ds = load_dataset("allenai/qasper", split="train")

rows = []
for paper in tqdm(qasper_ds, desc="Processing QASPER"):
    paper_id = paper["id"]
    title = paper["title"]
    sec_names = paper["full_text"]["section_name"]
    sec_paras = paper["full_text"]["paragraphs"]
    full_text = "\n\n".join(f"{s}\n" + "\n".join(p) for s, p in zip(sec_names, sec_paras))
    rows.append({"paper_id": paper_id, "title": title, "text": full_text})

df = pd.DataFrame(rows)
out_path = "/mnt/Qasper/processed_qasper_df.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} QASPER papers to {out_path}")

# # --- Create ChromaDB collection ---
# print(" Building ChromaDB collection for QASPER...")
# chroma_client = chromadb.PersistentClient(path="/mnt/ChromaDb")
# collection_name = "qasper_data_collection"

# try:
#     chroma_client.delete_collection(collection_name)
#     print(f"Old collection '{collection_name}' deleted.")
# except Exception:
#     pass

# embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
# collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_fn)

# for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding QASPER"):
#     text = str(row["text"])
#     chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
#     ids = [f"{row['paper_id']}_{i}" for i in range(len(chunks))]
#     metadatas = [{"paper_id": row["paper_id"]} for _ in chunks]
#     collection.add(documents=chunks, ids=ids, metadatas=metadatas)

# print(f" ChromaDB QASPER collection '{collection_name}' ready at /mnt/ChromaDb")
