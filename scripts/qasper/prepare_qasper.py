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
for paper in qasper_ds:
    paper_id = paper["id"]
    title    = paper["title"]
    abstract = paper["abstract"]

    # full_text is a dict of columns
    sec_names = paper["full_text"]["section_name"]
    sec_paras = paper["full_text"]["paragraphs"]
    full_text = "\n\n".join(
        f"{sec}\n" + "\n".join(p) for sec, p in zip(sec_names, sec_paras)
    )

    qas = paper["qas"]
    n_questions = len(qas["question"])

    for i in range(n_questions):
        question_id   = qas["question_id"][i]
        question_text = qas["question"][i]
        nlp_bg        = qas["nlp_background"][i]
        topic_bg      = qas["topic_background"][i]
        paper_read    = qas["paper_read"][i]
        search_query  = qas["search_query"][i]
        question_writer = qas["question_writer"][i]

        # answers is ALSO a dict of parallel lists
        answers_block = qas["answers"][i]
        for ans, ann_id, worker_id in zip(
            answers_block["answer"],
            answers_block["annotation_id"],
            answers_block["worker_id"]
        ):
            rows.append({
                "paper_id"        : paper_id,
                "title"           : title,
                "abstract"        : abstract,
                "context"        : full_text,
                "question_id"     : question_id,
                "question"        : question_text,
                "nlp_background"  : nlp_bg,
                "topic_background": topic_bg,
                "paper_read"      : paper_read,
                "search_query"    : search_query,
                "question_writer" : question_writer,
                "annotation_id"   : ann_id,
                "worker_id"       : worker_id,
                "unanswerable"    : ans["unanswerable"],
                "yes_no"          : ans["yes_no"],
                "free_form_answer": ans["free_form_answer"],
                "extractive_spans": "; ".join(ans["extractive_spans"]),
                "evidence"        : "; ".join(ans["evidence"]),
                "highlighted_evidence": "; ".join(ans["highlighted_evidence"])
            })

df = pd.DataFrame(rows)
out_path = "/mnt/Qasper/processed_qasper_df.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} QASPER papers to {out_path}")

DATASET_QASPER = "Qasper"

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
    name="qasper_data_collection",
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

def ingest_qasper_to_chroma(qasper_df, chroma_collection, character_splitter, token_splitter):
    for paper_id, group in tqdm(qasper_df.groupby("paper_id")):
        full_text = str(group.iloc[0]["full_text"])
    
        char_chunks = character_splitter.split_text(full_text)
    
        token_chunks = []
        for chunk in char_chunks:
            token_chunks.extend(token_splitter.split_text(chunk))
    
        if not token_chunks:
            print(f"Skipping paper {paper_id}: no chunks produced.")
            continue
        
        ids = [f"{paper_id}_{i}" for i in range(len(token_chunks))]
        question_ids = group["question_id"].tolist()
        metadatas = [
            {
                "paper_id": paper_id,
                "question_ids": ",".join(question_ids),
            }
            for _ in token_chunks
        ]
    
        chroma_collection.add(
            documents=token_chunks,
            ids=ids,
            metadatas=metadatas
        )
    
    print("All papers processed and stored in Chroma.")


ingest_qasper_to_chroma(df, collection, character_splitter, token_splitter)

