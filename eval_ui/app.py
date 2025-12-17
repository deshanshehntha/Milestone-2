"""Streamlit UI"""


from __future__ import annotations

import os
import re
import json
import time
import signal
import pathlib
import subprocess
from dataclasses import dataclass

import pandas as pd
import streamlit as st


APP_TITLE = "Quick Mixed Evaluation UI"
SCRIPT_NAME = "quick_mixed_eval.py" 
DEFAULT_RESULTS_PATH = "/data/results_quick"



def _safe_int(s: str, default: int) -> int:
    try:
        return int(s)
    except Exception:
        return default


def _find_run_dirs(results_path: str) -> list[str]:
    p = pathlib.Path(results_path)
    if not p.exists():
        return []
    runs = [d for d in p.iterdir() if d.is_dir()]
    runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return [str(r) for r in runs]


def _read_csv_if_exists(run_dir: str) -> pd.DataFrame | None:
    csv_path = os.path.join(run_dir, "evaluation_results.csv")
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return None
    return None


def _default_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _clean_id(x: object) -> str:
    """Normalize ids coming from CSV so 'nan' doesn't get used as a real filter value."""
    s = "" if x is None else str(x)
    s = s.strip()
    if not s or s.lower() == "nan" or s.lower() == "none":
        return ""
    return s


@dataclass
class EvalConfig:
    chroma_db_path: str
    router_addr: str
    eval_collections: str
    per_collection_samples: int
    peek_limit: int
    top_k: int
    grpc_timeout_s: int
    results_path: str
    run_id: str
    json_logs: bool
    embed_model: str
    restrict_same_question: bool
    mix_seed: int
    keep_cols: str


def build_env(cfg: EvalConfig) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "CHROMA_DB_PATH": cfg.chroma_db_path,
            "ROUTER_ADDR": cfg.router_addr,
            "EVAL_COLLECTIONS": cfg.eval_collections,
            "PER_COLLECTION_SAMPLES": str(cfg.per_collection_samples),
            "PEEK_LIMIT": str(cfg.peek_limit),
            "TOP_K": str(cfg.top_k),
            "GRPC_TIMEOUT_S": str(cfg.grpc_timeout_s),
            "RESULTS_PATH": cfg.results_path,
            "RUN_ID": cfg.run_id,
            "JSON_LOGS": "1" if cfg.json_logs else "0",
            "EMBED_MODEL": cfg.embed_model,
            "RESTRICT_TO_SAME_QUESTION": "1" if cfg.restrict_same_question else "0",
            "MIX_SEED": str(cfg.mix_seed),
            "KEEP_COLS": cfg.keep_cols,
        }
    )
    return env


def parse_json_lines(text: str) -> list[dict]:
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out



def render_metrics(df: pd.DataFrame):
    if df.empty:
        st.info("No rows in CSV.")
        return

    em = df["exact_match"].mean() if "exact_match" in df.columns else None
    cos = df["answer_cosine_sim"].mean() if "answer_cosine_sim" in df.columns else None
    err_rate = df["predicted_answer"].eq("ERROR").mean() if "predicted_answer" in df.columns else None

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", f"{len(df):,}")
    with c2:
        st.metric("Exact Match (mean)", f"{em:.3f}" if em is not None else "—")
    with c3:
        st.metric("Answer Cosine (mean)", f"{cos:.3f}" if cos is not None else "—")

    if err_rate is not None and err_rate > 0:
        st.warning(f"Errors: {err_rate:.2%} of rows had predicted_answer=ERROR")


def render_breakdowns(df: pd.DataFrame):
    if df.empty:
        return

    st.subheader("Routing")
    c1, c2 = st.columns(2)
    with c1:
        if "routed_variant" in df.columns:
            st.bar_chart(df["routed_variant"].fillna("(null)").value_counts())
        else:
            st.write("No routed_variant column")
    with c2:
        if "routed_by" in df.columns:
            st.bar_chart(df["routed_by"].fillna("(null)").value_counts())
        else:
            st.write("No routed_by column")

    st.subheader("Dataset / Collection")
    c3, c4 = st.columns(2)
    with c3:
        if "dataset" in df.columns:
            st.bar_chart(df["dataset"].fillna("(null)").value_counts())
        else:
            st.write("No dataset column")
    with c4:
        if "collection" in df.columns:
            st.bar_chart(df["collection"].fillna("(null)").value_counts())
        else:
            st.write("No collection column")


def render_table(df: pd.DataFrame):
    st.subheader("Results Table")

    cols = st.columns(4)
    with cols[0]:
        q = st.text_input("Search question", value="")
    with cols[1]:
        routed = st.multiselect(
            "routed_variant",
            options=sorted(df["routed_variant"].dropna().unique().tolist()) if "routed_variant" in df.columns else [],
            default=[],
        )
    with cols[2]:
        dataset = st.multiselect(
            "dataset",
            options=sorted(df["dataset"].dropna().unique().tolist()) if "dataset" in df.columns else [],
            default=[],
        )
    with cols[3]:
        show_errors_only = st.checkbox("Errors only", value=False)

    view = df
    if q and "question" in view.columns:
        view = view[view["question"].astype(str).str.contains(re.escape(q), case=False, na=False)]
    if routed and "routed_variant" in view.columns:
        view = view[view["routed_variant"].isin(routed)]
    if dataset and "dataset" in view.columns:
        view = view[view["dataset"].isin(dataset)]
    if show_errors_only and "predicted_answer" in view.columns:
        view = view[view["predicted_answer"].astype(str).eq("ERROR")]

    st.dataframe(view, use_container_width=True, height=450)


def render_single_question_runner(cfg: EvalConfig, df: pd.DataFrame):
    st.header("Ask the router")

    if df.empty or "question" not in df.columns:
        st.info("Load a run with a 'question' column to use this.")
        return

    def _label(i: int) -> str:
        row = df.iloc[i]
        q = str(row.get("question", "") or "")
        col = str(row.get("collection", "") or "")
        ds = str(row.get("dataset", "") or "")
        qid = str(row.get("question_id", "") or "")
        q_short = (q[:120] + "…") if len(q) > 120 else q
        return f"[{i}] {ds or col} | qid={qid} | {q_short}"

    def _load_row_into_state(i: int):
        r = df.iloc[int(i)]
        st.session_state["ask_q"] = str(r.get("question", "") or "")
        st.session_state["ask_dataset"] = str(r.get("dataset", "") or "")
        st.session_state["ask_collection"] = str(r.get("collection", "") or "")
        st.session_state["ask_story"] = _clean_id(r.get("story_id", ""))
        st.session_state["ask_qid"] = _clean_id(r.get("question_id", ""))

    if "ask_idx" not in st.session_state:
        st.session_state["ask_idx"] = 0
        _load_row_into_state(0)

    idx = st.selectbox(
        "Select a question from this run",
        options=list(range(len(df))),
        format_func=_label,
        key="ask_idx",
    )

    colA, colB = st.columns([1, 3])
    with colA:
        if st.button("Load selected row", use_container_width=True):
            _load_row_into_state(idx)
    with colB:
        st.caption("Pick a row → click **Load selected row** → fields update. Then you can edit freely.")

    with st.form("ask_form", clear_on_submit=False):
        st.caption("You can edit before sending.")
        q_text = st.text_area("Question", value=st.session_state.get("ask_q", ""), height=120, key="ask_q")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            dataset_override = st.text_input(
                "dataset (metadata)",
                value=st.session_state.get("ask_dataset", ""),
                key="ask_dataset",
            )
        with c2:
            collection_override = st.text_input(
                "collection",
                value=st.session_state.get("ask_collection", ""),
                key="ask_collection",
            )
        with c3:
            story_id = st.text_input(
                "story_id (optional)",
                value=st.session_state.get("ask_story", ""),
                key="ask_story",
            )
        with c4:
            question_id = st.text_input(
                "question_id (optional)",
                value=st.session_state.get("ask_qid", ""),
                key="ask_qid",
            )

        st.write("Context build:")
        d1, d2, d3 = st.columns(3)
        with d1:
            top_k = st.number_input("TOP_K", min_value=1, max_value=50, value=int(cfg.top_k), step=1, key="ask_top_k")
        with d2:
            variant = st.selectbox("variant", options=["auto", "llama3", "qwen2_5", "mistral"], index=0, key="ask_variant")
        with d3:
            timeout_s = st.number_input(
                "GRPC_TIMEOUT_S",
                min_value=1,
                max_value=600,
                value=int(cfg.grpc_timeout_s),
                step=5,
                key="ask_timeout",
            )

        send = st.form_submit_button("Send to router", type="primary", use_container_width=True)

    if not send:
        return

    import grpc
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    import qa_pb2
    import qa_pb2_grpc

    try:
        client = chromadb.PersistentClient(
            path=cfg.chroma_db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=cfg.embed_model)
        collection = client.get_collection(name=collection_override, embedding_function=embedding_fn)

        st.caption(f"Using collection: {collection.name}")
        st.caption(f"Collection count: {collection.count()}")
    except Exception as e:
        st.error(f"Chroma error: {e}")
        return

    where = None
    if cfg.restrict_same_question:
        sid = _clean_id(story_id)
        qid = _clean_id(question_id)
        if sid:
            where = {"story_id": sid}
        elif qid:
            where = {"question_id": qid}

    try:
        t0 = time.time()
        results = collection.query(
            query_texts=[q_text],
            n_results=int(top_k),
            where=where,  
            include=["documents", "metadatas"],
        )
        retrieval_ms = (time.time() - t0) * 1000.0

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if len(docs) == 0 and where is not None:
            st.warning("0 hits with filter; retrying without filter...")
            t0b = time.time()
            results = collection.query(
                query_texts=[q_text],
                n_results=int(top_k),
                where=None,
                include=["documents", "metadatas"],
            )
            retrieval_ms = (time.time() - t0b) * 1000.0
            docs = results["documents"][0]
            metas = results["metadatas"][0]

        parts = []
        for d, m in zip(docs, metas):
            ci = m.get("chunk_index", "")
            sid2 = m.get("story_id", "")
            qid2 = m.get("question_id", "")
            parts.append(f"[chunk {ci}] story_id={sid2} question_id={qid2}\n{d}")

        ctx = "\n\n---\n\n".join(parts).strip()
    except Exception as e:
        st.error(f"Context build error: {e}")
        return

    md = [("dataset", (dataset_override or "").strip().lower() or "unknown")]
    md.append(("variant", variant))
    md.append(("retrieval_k", str(int(top_k))))

    t1 = time.time()
    try:
        channel = grpc.insecure_channel(cfg.router_addr)
        stub = qa_pb2_grpc.QAServerStub(channel)
        resp, call = stub.Answer.with_call(
            qa_pb2.Question(question=q_text, context=ctx),
            metadata=md,
            timeout=float(timeout_s),
        )
    except grpc.RpcError as e:
        st.error(f"gRPC failed: {e.code()} — {e.details()}")
        st.json({"router_addr": cfg.router_addr, "metadata_sent": md})
        return
    except Exception as e:
        st.error(f"Router call error: {e}")
        st.json({"router_addr": cfg.router_addr, "metadata_sent": md})
        return

    grpc_ms = (time.time() - t1) * 1000.0
    trail = dict(call.trailing_metadata() or [])
    routed_variant = trail.get("routed-variant", "")
    routed_by = trail.get("routed-by", "")

    st.subheader("Response")
    if (resp.answer or "").strip():
        st.write(resp.answer)
    else:
        st.warning("Router returned an empty answer string (resp.answer is empty).")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("local retrieval (ms)", f"{retrieval_ms:.1f}")
    with m2:
        st.metric("grpc elapsed (ms)", f"{grpc_ms:.1f}")
    with m3:
        st.metric("routed_variant", routed_variant or "(none)")
    with m4:
        st.metric("routed_by", routed_by or "(none)")

    with st.expander("Context sent to router"):
        st.caption(f"Retrieved chunks: {len(docs)} • Context size: {len(ctx):,} chars")
        st.caption("Preview (first 4,000 chars):")
        st.code((ctx[:4000] if ctx else ""), language="text")

        st.caption("Full context:")
        st.text_area("ctx_full", value=(ctx or ""), height=420, label_visibility="collapsed")

        st.download_button(
            "Download context.txt",
            data=(ctx or "").encode("utf-8", errors="replace"),
            file_name="context.txt",
            mime="text/plain",
        )

    with st.expander("Router timing + debug"):
        st.json(
            {
                "router_addr": cfg.router_addr,
                "metadata_sent": md,
                "trailing_metadata": trail,
                "confidence": float(getattr(resp, "confidence", 0.0)),
                "router_retrieval_ms": float(getattr(resp, "retrieval_ms", 0.0)),
                "inference_ms": float(getattr(resp, "inference_ms", 0.0)),
                "end_to_end_ms": float(getattr(resp, "end_to_end_ms", 0.0)),
                "context_chars": len(ctx),
            }
        )


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    script_exists = os.path.exists(SCRIPT_NAME)

    with st.sidebar:
        st.header("Run settings")

        chroma_db_path = st.text_input("CHROMA_DB_PATH", value=os.getenv("CHROMA_DB_PATH", "/chromadb"))

        router_addr = st.text_input("ROUTER_ADDR", value=os.getenv("ROUTER_ADDR", "router:50050"))

        eval_collections = st.text_input(
            "EVAL_COLLECTIONS (comma-separated; blank = all)",
            value=os.getenv("EVAL_COLLECTIONS", "hotpotqa,narrativeqa,pubmedqa"),
        )

        per_collection_samples = st.number_input(
            "PER_COLLECTION_SAMPLES",
            min_value=1,
            max_value=100000,
            value=_safe_int(os.getenv("PER_COLLECTION_SAMPLES", "300"), 300),
            step=50,
        )

        peek_limit = st.number_input(
            "PEEK_LIMIT",
            min_value=int(per_collection_samples),
            max_value=200000000,
            value=_safe_int(os.getenv("PEEK_LIMIT", "5000"), 5000),
            step=500,
        )

        top_k = st.number_input(
            "TOP_K",
            min_value=1,
            max_value=50,
            value=_safe_int(os.getenv("TOP_K", "6"), 6),
            step=1,
        )

        grpc_timeout_s = st.number_input(
            "GRPC_TIMEOUT_S",
            min_value=1,
            max_value=600,
            value=_safe_int(os.getenv("GRPC_TIMEOUT_S", "60"), 60),
            step=5,
        )

        results_path = st.text_input("RESULTS_PATH", value=os.getenv("RESULTS_PATH", DEFAULT_RESULTS_PATH))

        run_id = st.text_input(
            "RUN_ID (blank = timestamp)",
            value=os.getenv("RUN_ID", ""),
            help="If blank, app uses a timestamp so you never overwrite old runs.",
        ).strip()
        if not run_id:
            run_id = _default_run_id()

        embed_model = st.text_input("EMBED_MODEL", value=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
        restrict_same_question = st.checkbox(
            "RESTRICT_TO_SAME_QUESTION",
            value=(os.getenv("RESTRICT_TO_SAME_QUESTION", "1") == "1"),
        )
        json_logs = st.checkbox("JSON_LOGS", value=(os.getenv("JSON_LOGS", "1") == "1"))

        mix_seed = st.number_input(
            "MIX_SEED",
            min_value=0,
            max_value=2**31 - 1,
            value=_safe_int(os.getenv("MIX_SEED", "42"), 42),
            step=1,
        )

        keep_cols = st.text_input(
            "KEEP_COLS (optional)",
            value=os.getenv("KEEP_COLS", ""),
            help="Comma-separated list of output columns to keep.",
        )

        cfg = EvalConfig(
            chroma_db_path=chroma_db_path,
            router_addr=router_addr,
            eval_collections=eval_collections,
            per_collection_samples=int(per_collection_samples),
            peek_limit=int(peek_limit),
            top_k=int(top_k),
            grpc_timeout_s=int(grpc_timeout_s),
            results_path=results_path,
            run_id=run_id,
            json_logs=bool(json_logs),
            embed_model=embed_model,
            restrict_same_question=bool(restrict_same_question),
            mix_seed=int(mix_seed),
            keep_cols=keep_cols,
        )

        st.divider()
        if script_exists:
            run_clicked = st.button("▶ Run evaluation", use_container_width=True)
        else:
            run_clicked = False
            st.info(
                f"Batch eval runner disabled (missing {SCRIPT_NAME}).\n"
                "Use your evaluate-quick container to generate CSVs, then view them here."
            )

    log_box = st.empty()
    progress = st.progress(0)

    if run_clicked:
        env = build_env(cfg)
        os.makedirs(cfg.results_path, exist_ok=True)

        st.info(f"Running: {SCRIPT_NAME} (RUN_ID={cfg.run_id})")

        cmd = ["python", SCRIPT_NAME]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )

        lines: list[str] = []
        json_events: list[dict] = []

        try:
            while True:
                if proc.stdout is None:
                    break

                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.05)
                    continue

                lines.append(line.rstrip("\n"))
                log_box.code("\n".join(lines[-200:]), language="text")

                if cfg.json_logs:
                    json_events.extend(parse_json_lines(line))
                    qa_count = sum(1 for e in json_events if e.get("event") == "qa_auto")
                    progress.progress(min(1.0, 0.05 + (qa_count % 200) / 220.0))

            rc = proc.wait()
            progress.progress(1.0)

            if rc != 0:
                st.error(f"Evaluation exited with code {rc}")
            else:
                st.success("Evaluation finished")

        except KeyboardInterrupt:
            st.warning("Interrupted")
            proc.send_signal(signal.SIGINT)

    st.divider()

    st.header("Results")
    run_dirs = _find_run_dirs(cfg.results_path)
    selected = st.selectbox(
        "Select a run",
        options=run_dirs,
        index=0 if run_dirs else None,
        format_func=lambda p: os.path.basename(p),
        key="run_select",
    )

    if selected:
        df = _read_csv_if_exists(selected)
        if df is None:
            st.info("No evaluation_results.csv found in this run yet.")
        else:
            render_metrics(df)

            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"evaluation_results_{os.path.basename(selected)}.csv",
                mime="text/csv",
            )

            render_breakdowns(df)
            render_single_question_runner(cfg, df)
            render_table(df)

            st.subheader("Raw files")
            st.write(f"Run directory: {selected}")
            if os.path.exists(os.path.join(selected, "evaluation_results.csv")):
                st.write("- evaluation_results.csv")


if __name__ == "__main__":
    main()
