"""Ring 4: duckdb-tantivy v0.4.0 recheck — persistent named indexes + native Chinese.

What changed vs the October run (v0.2.0):
  - v0.2.1+ ships jieba INSIDE the extension (language := 'chinese') — no external
    pre-tokenization needed
  - v0.3.0 persistent named indexes: tantivy_create_index builds an on-disk index once,
    tantivy_search_index queries it (no per-call rebuild)
  - Consequence for this bench: the ext's Chinese config is now feasible on the FULL
    3,999-query CmedqaRetrieval set (October: 150-query sample, 4.3 s/query ephemeral)

Requires the v0.4.0 release artifact (unsigned — the community registry still serves
v0.2.0 for duckdb 1.5.6 while PR#3593 is in review):

  curl -L -o tantivy.duckdb_extension \
    https://github.com/alitrack/duckdb-tantivy/releases/download/v0.4.0/tantivy-linux_amd64.duckdb_extension

Runs (arg: en | zh | all):
  en — SciFact: persistent vs ephemeral + bit-identity check over all 300 queries
  zh — CmedqaRetrieval FULL: persistent native-jieba OR / AND, ephemeral 150-sample
"""
import json, math, statistics as st, sys, time, os, re
from pathlib import Path

ROOT = Path("/tmp/swe")
EXT = os.environ.get("TANTIVY_EXT", "/tmp/tantivy.duckdb_extension")
SG = ROOT / "data" / "scifact"
CM = ROOT / "data" / "cmedqa"
(SG / "qrels").mkdir(exist_ok=True)
if (SG / "test.tsv").exists() and not (SG / "qrels" / "test.tsv").exists():
    os.replace(SG / "test.tsv", SG / "qrels" / "test.tsv")

import duckdb
import pyarrow.parquet as pq

def conn():
    c = duckdb.connect(config={"allow_unsigned_extensions": True})
    c.execute(f"LOAD '{EXT}'")
    return c

# ---------- data ----------
def load_scifact():
    docs = [dict(_id=r["_id"], title=r.get("title") or "", text=r["text"])
            for r in pq.read_table(SG / "corpus.parquet").to_pylist()]
    queries = {r["_id"]: r["text"] for r in pq.read_table(SG / "queries.parquet").to_pylist()}
    qrels = {}
    for line in (SG / "qrels" / "test.tsv").read_text().splitlines()[1:]:
        q, d, s = line.split("\t")
        if int(s) > 0:
            qrels.setdefault(q, {})[d] = int(s)
    qs = [q for q in qrels if q in queries]
    return docs, queries, qrels, qs

def load_cmedqa():
    corpus = pq.read_table(CM / "corpus.parquet").to_pylist()
    queries = {r["id"]: r["text"] for r in pq.read_table(CM / "queries.parquet").to_pylist()}
    qrels = {}
    for r in pq.read_table(CM / "qrels.parquet").to_pylist():
        if int(r["score"]) > 0:
            qrels.setdefault(str(r["qid"]), {})[str(r["pid"])] = int(r["score"])
    qs = [q for q in qrels if q in queries]
    return corpus, queries, qrels, qs

WORD = re.compile(r"[A-Za-z0-9]+")
def terms_en(q):
    return [t.lower() for t in WORD.findall(q)]

def ndcg10(ranked, rel):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    return dcg / sum(g / math.log2(i + 2) for i, g in enumerate(ideal)) if ideal else 0.0

def recall100(ranked, rel):
    return len(set(ranked[:100]) & set(rel)) / len(rel)

def dir_size(p):
    p = Path(p)
    return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

def bench(c, search_fn, queries, qs, qrels, label, id_map=None, build_s=-1.0, size=-1):
    for q in qs[:20]:
        search_fn(queries[q])
    lat, nd, rc, empty = [], [], [], 0
    for q in qs:
        t0 = time.perf_counter(); r = search_fn(queries[q]); lat.append(time.perf_counter() - t0)
        if id_map is not None:
            r = [id_map[i] for i in r]
        nd.append(ndcg10(r, qrels[q])); rc.append(recall100(r, qrels[q])); empty += not r
    lat.sort()
    rec = {"engine": label, "ndcg10": round(st.mean(nd), 4), "recall100": round(st.mean(rc), 4),
           "empty_results": empty, "p50_ms": round(1000 * lat[len(lat)//2], 2),
           "p95_ms": round(1000 * lat[int(len(lat)*.95)], 2),
           "build_s": round(build_s, 2), "index_mb": round(size / 2**20, 1)}
    print(json.dumps(rec), flush=True)
    return rec

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = []

    # ============ English ============
    if which in ("all", "en"):
        docs, queries, qrels, qs = load_scifact()
        c = conn()
        c.execute("create table docs (doc_id bigint, body varchar)")
        t0 = time.perf_counter()
        c.executemany("insert into docs values (?,?)",
                      [(i, (d["title"] + " . " + d["text"])) for i, d in enumerate(docs)])
        ingest = time.perf_counter() - t0
        id_map = {i: d["_id"] for i, d in enumerate(docs)}

        # persistent index
        t0 = time.perf_counter()
        c.execute("SELECT * FROM tantivy_create_index('sf', 'docs', id_column := 'doc_id', language := 'english')")
        build_s = time.perf_counter() - t0
        idx_dir = Path(".tantivy_index/sf")
        size = dir_size(idx_dir) if idx_dir.exists() else -1
        def s_persist(q, k=100):
            ts = " OR ".join(terms_en(q))
            return [r[0] for r in c.execute(
                "select doc_id from tantivy_search_index('sf', ?, top_k := ?)", [ts, k]).fetchall()]
        results.append(bench(c, s_persist, queries, qs, qrels,
                             "duckdb-tantivy v0.4.0 persistent (en)", id_map, build_s, size))
        print("ingest_s", round(ingest, 2), flush=True)

        # ephemeral bm25_search (v0.4.0)
        def s_ephem(q, k=100):
            ts = " OR ".join(terms_en(q))
            return [r[0] for r in c.execute(
                "select doc_id from bm25_search('docs', ?, top_k := ?)", [ts, k]).fetchall()]
        results.append(bench(c, s_ephem, queries, qs, qrels,
                             "duckdb-tantivy v0.4.0 ephemeral (en)", id_map))

        # bit identity persistent vs ephemeral
        diff = 0
        for q in qs:
            a = s_persist(queries[q]); b = s_ephem(queries[q])
            if a != b:
                diff += 1
        print(json.dumps({"check": "bit-identity persist-vs-ephemeral (en)", "queries": len(qs), "diffs": diff}), flush=True)
        results.append({"engine": "bit-identity persist vs ephemeral (en)", "queries": len(qs), "diffs": diff})

    # ============ Chinese ============
    if which in ("all", "zh"):
        corpus, queries, qrels, qs = load_cmedqa()
        print(f"docs={len(corpus)} queries={len(qs)}", flush=True)
        c = conn()
        c.execute("create table docs (doc_id bigint, body varchar)")
        t0 = time.perf_counter()
        import pyarrow as pa
        tbl = pa.table({
            "doc_id": pa.array(list(range(len(corpus))), type=pa.int64()),
            "body": pa.array([r["text"] for r in corpus], type=pa.string()),
        })
        c.register("bulk", tbl)
        c.execute("insert into docs select doc_id, body from bulk")
        ingest = time.perf_counter() - t0
        id_map = {i: r["id"] for i, r in enumerate(corpus)}
        print("ingest_s", round(ingest, 2), flush=True)

        # (a) persistent, native chinese, OR queries
        t0 = time.perf_counter()
        c.execute("SELECT * FROM tantivy_create_index('cm', 'docs', id_column := 'doc_id', language := 'chinese')")
        build_s = time.perf_counter() - t0
        idx_dir = Path(".tantivy_index/cm")
        size = dir_size(idx_dir) if idx_dir.exists() else -1
        print("zh index build_s", round(build_s, 2), "size_mb", round(size / 2**20, 1), flush=True)

        import jieba
        jieba.initialize()
        def cut_q(q):
            return " OR ".join(f'"{t}"' for t in jieba.cut_for_search(q) if re.search(r"[\w\u4e00-\u9fff]", t))
        def s_zh_or(q, k=100):
            return [r[0] for r in c.execute(
                "select doc_id from tantivy_search_index('cm', ?, top_k := ?)", [cut_q(q), k]).fetchall()]
        results.append(bench(c, s_zh_or, queries, qs, qrels,
                             "duckdb-tantivy v0.4.0 persistent zh native-jieba OR (full 3999)",
                             id_map, build_s, size))

        # (b) persistent, native chinese, AND (v0.4.0 default semantics, raw query)
        def s_zh_and(q, k=100):
            safe = "".join(ch if re.search(r"[\w\u4e00-\u9fff\s]", ch) else " " for ch in q)
            return [r[0] for r in c.execute(
                "select doc_id from tantivy_search_index('cm', ?, top_k := ?)", [safe, k]).fetchall()]
        results.append(bench(c, s_zh_and, queries, qs, qrels,
                             "duckdb-tantivy v0.4.0 persistent zh native-jieba AND (full 3999)",
                             id_map))

        # (c) ephemeral native chinese on a 150-query sample (same seed as the October run)
        import random
        rng = random.Random(42)
        sample = sorted(rng.sample(qs, 150))
        def s_zh_eph(q, k=100):
            return [r[0] for r in c.execute(
                "select doc_id from bm25_search('docs', ?, top_k := ?, language := 'chinese')",
                [cut_q(q), k]).fetchall()]
        lat, nd, rc, empty = [], [], [], 0
        for q in sample:
            t0 = time.perf_counter(); r = s_zh_eph(queries[q]); lat.append(time.perf_counter() - t0)
            r = [id_map[i] for i in r]
            nd.append(ndcg10(r, qrels[q])); rc.append(recall100(r, qrels[q])); empty += not r
        lat.sort()
        results.append({"engine": "duckdb-tantivy v0.4.0 ephemeral zh native-jieba OR (n=150)",
                        "ndcg10": round(st.mean(nd), 4), "recall100": round(st.mean(rc), 4),
                        "empty_results": empty, "p50_ms": round(1000 * lat[len(lat)//2], 2),
                        "p95_ms": round(1000 * lat[int(len(lat)*.95)], 2), "build_s": -1, "index_mb": -1})
        print(json.dumps(results[-1]), flush=True)

    (ROOT / f"results_v040_{which}.json").write_text(json.dumps(results, indent=1))
    print("saved", f"results_v040_{which}.json")

if __name__ == "__main__":
    main()
