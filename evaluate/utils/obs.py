# evaluate/utils/obs.py
import json, time, hashlib

def now_ms() -> int:
    return int(time.time() * 1000)

def qhash(s: str) -> str:
    return hashlib.sha256(s.strip().lower().encode("utf-8")).hexdigest()[:12]

def log_event(**kv):
    print(json.dumps(kv, ensure_ascii=False), flush=True)

def norm_q(s: str) -> str:
    return " ".join(s.strip().lower().split())
