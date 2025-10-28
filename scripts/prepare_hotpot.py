import os
import json
import pandas as pd
import kagglehub

os.makedirs("/mnt/HotpotQA", exist_ok=True)

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

