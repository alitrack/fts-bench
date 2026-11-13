#!/usr/bin/env python3
"""Fetch bench datasets (run once, ~65 MB total).

Sources (as of 2026-11): the beir-cellar/beir GitHub repo no longer hosts dataset
files (404) — use the HF mirrors instead:

  SciFact          BeIR/scifact (corpus/queries parquet) + BeIR/scifact-qrels (test.tsv)
  CmedqaRetrieval  C-MTEB/CmedqaRetrieval (corpus/queries parquet)
                   + C-MTEB/CmedqaRetrieval-qrels (dev parquet, 3,999 queries)

hf-mirror.com is used by default (set HF_HOST=https://huggingface.co to override).
Layout produced (matches what the harnesses expect):
  data/scifact/corpus.parquet  data/scifact/queries.parquet  data/scifact/qrels/test.tsv
  data/cmedqa/corpus.parquet    data/cmedqa/queries.parquet   data/cmedqa/qrels.parquet
"""
import json, os, sys, time, urllib.request
from pathlib import Path

HOST = os.environ.get("HF_HOST", "https://hf-mirror.com")
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/swe/data")
SG = ROOT / "scifact"; (SG / "qrels").mkdir(parents=True, exist_ok=True)
CM = ROOT / "cmedqa"; CM.mkdir(parents=True, exist_ok=True)

FILES = [
    (f"{HOST}/datasets/BeIR/scifact/resolve/main/corpus/corpus-00000-of-00001.parquet",
     SG / "corpus.parquet"),
    (f"{HOST}/datasets/BeIR/scifact/resolve/main/queries/queries-00000-of-00001.parquet",
     SG / "queries.parquet"),
    (f"{HOST}/datasets/BeIR/scifact-qrels/resolve/main/test.tsv",
     SG / "qrels" / "test.tsv"),
    (f"{HOST}/datasets/C-MTEB/CmedqaRetrieval/resolve/main/data/corpus-00000-of-00001-a3949861f65a3226.parquet",
     CM / "corpus.parquet"),
    (f"{HOST}/datasets/C-MTEB/CmedqaRetrieval/resolve/main/data/queries-00000-of-00001-daeedab899d3c839.parquet",
     CM / "queries.parquet"),
    (f"{HOST}/datasets/C-MTEB/CmedqaRetrieval-qrels/resolve/main/data/dev-00000-of-00001-57fb84a4aceaa695.parquet",
     CM / "qrels.parquet"),
]

def get(url, dest, tries=4):
    if dest.exists() and dest.stat().st_size > 0:
        print("skip", dest); return
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fts-bench/0.2"})
            with urllib.request.urlopen(req, timeout=180) as r, open(str(dest) + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(str(dest) + ".part", dest)
            print("ok", dest, dest.stat().st_size); return
        except Exception as e:
            last = e; print("retry", i, e); time.sleep(3 * (i + 1))
    raise SystemExit(f"failed: {url}: {last}")

if __name__ == "__main__":
    for url, dest in FILES:
        get(url, dest)
    print("ALL DONE")
