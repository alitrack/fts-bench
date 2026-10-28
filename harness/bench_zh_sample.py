"""Subsample run: DuckDB tantivy ext + Postgres on 150 Chinese queries (ext rebuilds index per call — full 3999 infeasible)."""
import json, math, statistics as st, sys, time, os, re, random
from pathlib import Path
ROOT = Path("/tmp/swe")
sys.path.insert(0, str(ROOT / "harness"))
sys.argv = ["bench_zh.py", "__none__"]  # neutralize module-level engine dispatch
import bench_zh as B  # reuse loaders & classes

docs = B.docs; queries = B.queries; qrels = B.qrels
qs = [q for q in qrels if q in queries]
random.seed(42)
sample = random.sample(qs, 150)
out = []
# DuckDB tantivy ext
e = B.DuckTantivyZH()
e.build(docs)
for q in sample[:3]: e.search(queries[q])
lat, nd, rc = [], [], []
for q in sample:
    t0 = time.perf_counter(); r = e.search(queries[q]); lat.append(time.perf_counter()-t0)
    nd.append(B.ndcg10(r, qrels[q])); rc.append(B.recall100(r, qrels[q]))
lat.sort()
out.append({"engine": "DuckDB+tantivy ext + jieba (n=150)", "ndcg10": round(st.mean(nd),4), "recall100": round(st.mean(rc),4),
            "p50_ms": round(1000*lat[len(lat)//2],2), "p95_ms": round(1000*lat[int(len(lat)*.95)],2)})
print(json.dumps(out[-1]), flush=True)
# Postgres
e = B.PostgresZH()
e.build(docs)
for q in sample[:20]: e.search(queries[q])
lat, nd, rc = [], [], []
for q in sample:
    t0 = time.perf_counter(); r = e.search(queries[q]); lat.append(time.perf_counter()-t0)
    nd.append(B.ndcg10(r, qrels[q])); rc.append(B.recall100(r, qrels[q]))
lat.sort()
out.append({"engine": "Postgres simple + jieba (n=150)", "ndcg10": round(st.mean(nd),4), "recall100": round(st.mean(rc),4),
            "p50_ms": round(1000*lat[len(lat)//2],2), "p95_ms": round(1000*lat[int(len(lat)*.95)],2)})
print(json.dumps(out[-1]), flush=True)
(ROOT / "results_zh_sample.json").write_text(json.dumps(out, indent=1))
print("saved")
