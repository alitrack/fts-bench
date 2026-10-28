"""Ring 1 (English reproduction, SciFact-only, local mode).

Reproduces nazmul284/search-without-elasticsearch on this machine:
  Postgres (17, local cluster), SQLite FTS5, DuckDB FTS, tantivy-py,
  + extra candidates: DuckDB tantivy community extension (alitrack),
    DuckDB built-in FTS config variants, Tantivy raw (no stem).
Uses the repo's own harness semantics: build once per engine, OR-mode
queries (any term may match), nDCG@10 + recall@100 vs BEIR qrels.
"""
import json, math, statistics as st, sys, time, os, re
from pathlib import Path

ROOT = Path("/tmp/swe")
D = ROOT / "data" / "scifact"
W = ROOT / "data" / "work"; W.mkdir(exist_ok=True)
docs = [json.loads(l) for l in (D / "corpus.jsonl").open()]
queries = {j["_id"]: j["text"] for j in map(json.loads, (D / "queries.jsonl").open())}
qrels = {}
for line in list((D / "qrels" / "test.tsv").open())[1:]:
    q, d, s = line.rstrip("\n").split("\t")
    if int(s) > 0:
        qrels.setdefault(q, {})[d] = int(s)
qs = [q for q in qrels if q in queries]

WORD = re.compile(r"[A-Za-z0-9]+")
def terms(q):
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

def bench(e, label):
    e.build(docs)
    for q in qs[:20]:
        e.search(queries[q])  # warm-up
    lat, nd, rc, empty = [], [], [], 0
    for q in qs:
        t0 = time.perf_counter(); r = e.search(queries[q]); lat.append(time.perf_counter() - t0)
        nd.append(ndcg10(r, qrels[q])); rc.append(recall100(r, qrels[q])); empty += not r
    lat.sort()
    rec = {"engine": label, "ndcg10": round(st.mean(nd), 4), "recall100": round(st.mean(rc), 4),
           "empty_results": empty, "p50_ms": round(1000 * lat[len(lat)//2], 2),
           "p95_ms": round(1000 * lat[int(len(lat)*.95)], 2),
           "build_s": round(getattr(e, "_build_s", -1), 2), "index_mb": round(getattr(e, "_size", -1) / 2**20, 1)}
    print(json.dumps(rec), flush=True)
    return rec

class SQLiteFTS5:
    name = "SQLite FTS5 (porter)"
    def __init__(self, path, tokenize="porter unicode61", label=None):
        self.path, self.tokenize, self.label = str(path), tokenize, label or f"SQLite FTS5 ({tokenize})"
    def build(self, docs):
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(self.path + suffix): os.remove(self.path + suffix)
        import sqlite3
        self.c = sqlite3.connect(self.path)
        self.c.execute(f"create virtual table docs using fts5(id unindexed, title, body, tokenize='{self.tokenize}')")
        t0 = time.perf_counter()
        self.c.executemany("insert into docs values (?,?,?)", ((d["_id"], d.get("title",""), d["text"]) for d in docs))
        self.c.execute("insert into docs(docs) values('optimize')"); self.c.commit()
        self._build_s = time.perf_counter() - t0; self._size = dir_size(self.path)
    def search(self, q, k=100):
        ts = ['"'+t+'"' for t in terms(q)]
        if not ts: return []
        expr = " OR ".join(ts)
        return [r[0] for r in self.c.execute("select id from docs where docs match ? order by bm25(docs) limit ?", (expr, k))]

class DuckFTS:
    name = "DuckDB FTS (builtin)"
    def __init__(self, path, stemmer="english", stop="english", label=None):
        self.path, self.stemmer, self.stop, self.label = str(path), stemmer, stop, label or "DuckDB FTS"
    def build(self, docs):
        import duckdb
        for p in (self.path, self.path + ".wal"):
            if os.path.exists(p): os.remove(p)
        self.c = duckdb.connect(self.path)
        self.c.execute("install fts; load fts")
        self.c.execute("create table docs (id varchar, title varchar, body varchar)")
        t0 = time.perf_counter()
        self.c.executemany("insert into docs values (?,?,?)", [(d["_id"], d.get("title",""), d["text"]) for d in docs])
        self.c.execute(f"pragma create_fts_index('docs','id','title','body', stemmer='{self.stemmer}', stopwords='{self.stop}')")
        self.c.execute("checkpoint")
        self._build_s = time.perf_counter() - t0; self._size = dir_size(self.path)
    def search(self, q, k=100):
        return [r[0] for r in self.c.execute(
            "select id from (select id, fts_main_docs.match_bm25(id, ?, conjunctive := 0) as s from docs) where s is not null order by s desc limit ?",
            [q, k]).fetchall()]

class TantivyPy:
    def __init__(self, path, tokenizer="en_stem", label=None):
        self.path, self.tokenizer, self.label = str(path), tokenizer, label or f"Tantivy ({tokenizer})"
    def build(self, docs):
        import tantivy, shutil
        shutil.rmtree(self.path, ignore_errors=True); os.makedirs(self.path)
        sb = tantivy.SchemaBuilder()
        sb.add_text_field("id", stored=True, tokenizer_name="raw")
        sb.add_text_field("title", tokenizer_name=self.tokenizer)
        sb.add_text_field("body", tokenizer_name=self.tokenizer)
        self.index = tantivy.Index(sb.build(), path=self.path)
        t0 = time.perf_counter()
        w = self.index.writer(heap_size=256_000_000)
        for d in docs:
            w.add_document(tantivy.Document(id=d["_id"], title=d.get("title",""), body=d["text"]))
        w.commit(); w.wait_merging_threads(); self.index.reload()
        self.searcher = self.index.searcher()
        self._build_s = time.perf_counter() - t0; self._size = dir_size(self.path)
    def search(self, q, k=100):
        ts = terms(q)
        if not ts: return []
        query = self.index.parse_query(" ".join(ts), ["title", "body"])
        return [self.searcher.doc(a)["id"][0] for _, a in self.searcher.search(query, k).hits]

class DuckTantivy:
    """alitrack/duckdb-tantivy community extension: bm25_search() over a table."""
    name = "DuckDB+tantivy ext"
    def __init__(self, label="DuckDB tantivy ext"):
        self.label = label
    def build(self, docs):
        import duckdb
        self.c = duckdb.connect()
        self.c.execute("install tantivy from community; load tantivy")
        self.c.execute("create table docs (doc_id bigint, body varchar)")
        self.c.execute("create table ids (rowno bigint, doc_id varchar)")
        t0 = time.perf_counter()
        self.c.executemany("insert into docs values (?,?)",
                           [(i, (d.get("title","") + " . " + d["text"])) for i, d in enumerate(docs)])
        self.c.executemany("insert into ids values (?,?)",
                           [(i, d["_id"]) for i, d in enumerate(docs)])
        self.id_map = {i: d["_id"] for i, d in enumerate(docs)}
        self._ingest_s = time.perf_counter() - t0
        self._build_s = self._ingest_s  # index is built per call
        self._size = 0
    def search(self, q, k=100):
        ts = terms(q)
        if not ts: return []
        expr = " OR ".join(ts)  # tantivy parser default multi-term is AND; repo compares OR mode
        rows = self.c.execute(
            "select doc_id from bm25_search('docs', ?, top_k := ?)", [expr, k]).fetchall()
        return [self.id_map[r[0]] for r in rows]  # bm25_search rows are score-ordered

class PostgresFTS:
    name = "Postgres"
    DSN = "host=localhost port=5432 user=bench password=" + os.environ.get("PGPASSWORD", "bench") + " dbname=postgres"
    def __init__(self, port=5432, label="Postgres 17"):
        self.port, self.label = port, label
    def build(self, docs):
        import psycopg
        with psycopg.connect(self.DSN, autocommit=True) as c:
            c.execute("drop database if exists search_bench")
            c.execute("create database search_bench")
        self.c = psycopg.connect(self.DSN.replace("dbname=postgres", "dbname=search_bench"), autocommit=True)
        c = self.c
        c.execute("""create table docs (id text primary key, title text, body text,
            tsv tsvector generated always as (setweight(to_tsvector('english', coalesce(title,'')),'A') ||
                                              setweight(to_tsvector('english', coalesce(body,'')),'B')) stored)""")
        t0 = time.perf_counter()
        with c.cursor() as cur, cur.copy("copy docs (id, title, body) from stdin") as cp:
            for d in docs:
                cp.write_row((d["_id"], d.get("title",""), d["text"]))
        c.execute("create index docs_tsv on docs using gin (tsv)")
        c.execute("vacuum analyze docs")
        self._build_s = time.perf_counter() - t0
        self._size = c.execute("select pg_total_relation_size('docs')").fetchone()[0]
    def search(self, q, k=100):
        ts = " | ".join(terms(q)) or "x"
        sql = ("select id from docs, to_tsquery('english', %s) query where tsv @@ query "
               "order by ts_rank(tsv, query) desc limit %s")
        return [r[0] for r in self.c.execute(sql, (ts, k)).fetchall()]

results = []
which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("all", "core"):
    results.append(bench(SQLiteFTS5(W/"scifact.sqlite"), "SQLite FTS5 porter (repo config)"))
    results.append(bench(DuckFTS(W/"scifact.duckdb"), "DuckDB FTS builtin (repo config)"))
    results.append(bench(TantivyPy(W/"scifact.tantivy"), "Tantivy en_stem (repo config)"))
    results.append(bench(PostgresFTS(), "Postgres 17 (repo config)"))
if which in ("all", "ext"):
    results.append(bench(DuckTantivy(), "DuckDB+tantivy community ext"))
if which in ("all", "tune"):
    results.append(bench(TantivyPy(W/"scifact.raw.tantivy", tokenizer="raw", label="Tantivy raw (no stem)"), "Tantivy raw (no stem)"))
    results.append(bench(DuckFTS(W/"scifact-nostop.duckdb", stop="none", label="DuckDB FTS nostop"), "DuckDB FTS no-stopwords"))
    results.append(bench(SQLiteFTS5(W/"scifact-uni.sqlite", tokenize="unicode61", label="SQLite FTS5 unicode61"), "SQLite FTS5 unicode61 (no porter)"))
(ROOT / f"results_local_{which}.json").write_text(json.dumps(results, indent=1))
print("saved", f"results_local_{which}.json")
