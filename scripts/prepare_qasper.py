import os
import pandas as pd
from datasets import load_dataset

# Make output folder
os.makedirs("/mnt/Qasper", exist_ok=True)

# Load the QASPER dataset
print("📦 Downloading allenai/qasper dataset ...")
ds = load_dataset("allenai/qasper", split="train")

rows = []
for paper in ds:
    if paper["qas"]["question"]:
        rows.append({
            "id": paper["id"],
            "title": paper["title"],
            "n_questions": len(paper["qas"]["question"])
        })

df = pd.DataFrame(rows)
out_path = "/mnt/Qasper/summary.csv"
df.to_csv(out_path, index=False)

print(f"✅ Saved {out_path} with {len(df)} papers")
