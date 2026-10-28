"""duckdb-luajit route: jieba tokenization INSIDE DuckDB, then built-in FTS + tantivy ext.
Full 100k corpus this time (probe passed)."""
import json, math, statistics as st, sys, time, re
from pathlib import Path
ROOT = Path("/tmp/swe")
sys.path.insert(0, str(ROOT / "harness"))
sys.argv = ["bench_zh.py", "__none__"]
import bench_zh as B

docs = B.docs; queries = B.queries; qrels = B.qrels
qs = [q for q in qrels if q in queries]
print(f"docs={len(docs)} queries={len(qs)}", flush=True)

import duckdb

# ---- Route A: luajit jieba -> built-in FTS (persistent index) ----
c = duckdb.connect(str(B.W / "zh_luajit.duckdb"))
c.execute("install luajit from community; load luajit")
c.execute("select * from luajit_module(mode := 'install', sql_name := 'jieba')").fetchall()
c.execute("install fts; load fts")
c.execute("create or replace table docs (id varchar, body varchar)")
t0 = time.perf_counter()
c.executemany("insert into docs select ?, ((luajit_s('jieba', {v: ?, op: 'join', sep: ' '}) ->> 'joined'))",
              [(d["_id"], d["title"] + " " + d["text"]) for d in docs])
n_tok = time.perf_counter() - t0
print("tokenize+insert:", round(n_tok, 1), "s", flush=True)
t1 = time.perf_counter()
c.execute("pragma create_fts_index('docs','id','body')")
c.execute("checkpoint")
print("fts index:", round(time.perf_counter()-t1, 1), "s", flush=True)

def search_luajit(q, k=100):
    jq = c.execute("select luajit_s('jieba', {v: ?, op: 'join', sep: ' '}) ->> 'joined'", [q]).fetchone()[0]
    if not jq: return []
    return [r[0] for r in c.execute(
        "select id from (select id, fts_main_docs.match_bm25(id, ?, conjunctive := 0) as s from docs) where s is not null order by s desc limit ?",
        [jq, k]).fetchall()]

for q in qs[:20]: search_luajit(queries[q])
lat, nd, rc, empty = [], [], [], 0
for q in qs:
    t0 = time.perf_counter(); r = search_luajit(queries[q]); lat.append(time.perf_counter()-t0)
    nd.append(B.ndcg10(r, qrels[q])); rc.append(B.recall100(r, qrels[q])); empty += not r
lat.sort()
print(json.dumps({"engine": "DuckDB FTS + luajit-jieba (all-SQL)", "ndcg10": round(st.mean(nd),4), "recall100": round(st.mean(rc),4),
      "empty": empty, "p50_ms": round(1000*lat[len(lat)//2],2), "p95_ms": round(1000*lat[int(len(lat)*.95)],2),
      "tokenize_s": round(n_tok,1)}), flush=True)
