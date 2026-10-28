"""Ring 2 (Chinese): C-MTEB CmedqaRetrieval dev split, BM25 engines with Chinese tokenization.

Engines:
  SQLite FTS5  (a) trigram tokenizer  (b) unicode61 (single CJK char tokens? -> measures raw default)
  DuckDB FTS   (a) built-in default (no stemmer/stopwords; CJK handled how? measured)
  Tantivy-py   (a) jieba pre-tokenized + whitespace 'default' tokenizer
               (b) raw (untokenized CJK -> single long tokens)
  DuckDB tantivy community ext  (language param off; queries jieba-tokenized OR-joined)
  Postgres     to_tsvector('simple') on jieba-pre-tokenized text (space-joined), GIN, ts_rank
  (optional reference) SQLite FTS5 porter unicode61 on jieba text

OR semantics for multi-term queries (matching the English ring).
"""
import json, math, statistics as st, sys, time, os, re
from pathlib import Path

ROOT = Path("/tmp/swe")
D = ROOT / "data" / "cmedqa"
W = D / "work"; W.mkdir(exist_ok=True)

import jieba
jieba.initialize()

def cut_zh(text):
    return [t.strip() for t in jieba.cut_for_search(text) if t.strip()]

import pyarrow.parquet as pq
corpus = pq.ParquetFile(D / "corpus.parquet").read().to_pandas()
queries = pq.ParquetFile(D / "queries.parquet").read().to_pandas()
qrels_df = pq.ParquetFile(D / "qrels.parquet").read().to_pandas()
docs = [{"_id": r["_id"], "title": r.get("title", "") or "", "text": r["text"]} for _, r in corpus.iterrows()]
queries = {r["_id"]: r["text"] for _, r in queries.iterrows()}
qrels = {}
for _, r in qrels_df.iterrows():
    if int(r["score"]) > 0:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
qs = [q for q in qrels if q in queries]
print(f"docs={len(docs)} queries={len(qs)}", flush=True)

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
        e.search(queries[q])
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

class SQLiteZH:
    def __init__(self, path, tokenize, pretok, label):
        self.path, self.tokenize, self.pretok, self.label = str(path), tokenize, pretok, label
    def build(self, docs):
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(self.path + suffix): os.remove(self.path + suffix)
        import sqlite3
        self.c = sqlite3.connect(self.path)
        self.c.execute(f"create virtual table docs using fts5(id unindexed, body, tokenize='{self.tokenize}')")
        t0 = time.perf_counter()
        if self.pretok:
            self.c.executemany("insert into docs values (?,?)",
                ((d["_id"], " ".join(cut_zh(d["title"] + " " + d["text"]))) for d in docs))
        else:
            self.c.executemany("insert into docs values (?,?)",
                ((d["_id"], d["title"] + " " + d["text"]) for d in docs))
        self.c.execute("insert into docs(docs) values('optimize')"); self.c.commit()
        self._build_s = time.perf_counter() - t0; self._size = dir_size(self.path)
    def _qexpr(self, q):
        if self.pretok:
            toks = cut_zh(q)
        else:
            toks = re.findall(r"\S+", q)
        # fts5 query string escaping: quote each token, double embedded quotes
        ts = ['"'+t.replace('"','""')+'"' for t in toks]
        return " OR ".join(ts) if ts else ""
    def search(self, q, k=100):
        expr = self._qexpr(q)
        if not expr: return []
        return [r[0] for r in self.c.execute("select id from docs where docs match ? order by bm25(docs) limit ?", (expr, k))]

class DuckZH:
    def __init__(self, path, pretok, label, stemmer=None, stop=None):
        self.path, self.pretok, self.label, self.stemmer, self.stop = str(path), pretok, label, stemmer, stop
    def build(self, docs):
        import duckdb
        for p in (self.path, self.path + ".wal"):
            if os.path.exists(p): os.remove(p)
        self.c = duckdb.connect(self.path)
        self.c.execute("install fts; load fts")
        self.c.execute("create table docs (id varchar, body varchar)")
        t0 = time.perf_counter()
        if self.pretok:
            self.c.executemany("insert into docs values (?,?)",
                [(d["_id"], " ".join(cut_zh(d["title"] + " " + d["text"]))) for d in docs])
        else:
            self.c.executemany("insert into docs values (?,?)",
                [(d["_id"], d["title"] + " " + d["text"]) for d in docs])
        extra = ""
        if self.stemmer: extra += f", stemmer='{self.stemmer}'"
        if self.stop: extra += f", stopwords='{self.stop}'"
        self.c.execute(f"pragma create_fts_index('docs','id','body'{extra})")
        self.c.execute("checkpoint")
        self._build_s = time.perf_counter() - t0; self._size = dir_size(self.path)
    def search(self, q, k=100):
        if self.pretok:
            qq = " ".join(cut_zh(q))
        else:
            qq = q
        return [r[0] for r in self.c.execute(
            "select id from (select id, fts_main_docs.match_bm25(id, ?, conjunctive := 0) as s from docs) where s is not null order by s desc limit ?",
            [qq, k]).fetchall()]

class TantivyZH:
    def __init__(self, path, pretok, tokenizer="default", label="Tantivy"):
        self.path, self.pretok, self.tokenizer, self.label = str(path), pretok, tokenizer, label
    def build(self, docs):
        import tantivy, shutil
        shutil.rmtree(self.path, ignore_errors=True); os.makedirs(self.path)
        sb = tantivy.SchemaBuilder()
        sb.add_text_field("id", stored=True, tokenizer_name="raw")
        sb.add_text_field("body", tokenizer_name=self.tokenizer)
        self.index = tantivy.Index(sb.build(), path=self.path)
        t0 = time.perf_counter()
        w = self.index.writer(heap_size=256_000_000)
        if self.pretok:
            for d in docs:
                w.add_document(tantivy.Document(id=d["_id"], body=" ".join(cut_zh(d["title"] + " " + d["text"]))))
        else:
            for d in docs:
                w.add_document(tantivy.Document(id=d["_id"], body=d["title"] + " " + d["text"]))
        w.commit(); w.wait_merging_threads(); self.index.reload()
        self.searcher = self.index.searcher()
        self._build_s = time.perf_counter() - t0; self._size = dir_size(self.path)
    def search(self, q, k=100):
        ts = cut_zh(q) if self.pretok else re.findall(r"\S+", q)
        if not ts: return []
        safe = ['"'+t.replace('"','')+'"' for t in ts if re.search(r"[\w\u4e00-\u9fff]", t)]
        if not safe: return []
        query = self.index.parse_query(" OR ".join(safe), ["body"])
        return [self.searcher.doc(a)["id"][0] for _, a in self.searcher.search(query, k).hits]

class DuckTantivyZH:
    def __init__(self, label="DuckDB+tantivy ext"):
        self.label = label
    def build(self, docs):
        import duckdb
        self.c = duckdb.connect()
        self.c.execute("install tantivy from community; load tantivy")
        self.c.execute("create table docs (doc_id bigint, body varchar)")
        t0 = time.perf_counter()
        self.c.executemany("insert into docs values (?,?)",
            [(i, " ".join(cut_zh(d["title"] + " " + d["text"]))) for i, d in enumerate(docs)])
        self.id_map = {i: d["_id"] for i, d in enumerate(docs)}
        self._build_s = time.perf_counter() - t0; self._size = 0
    def search(self, q, k=100):
        ts = [t for t in cut_zh(q) if re.search(r"[\w\u4e00-\u9fff]", t)]
        if not ts: return []
        expr = " OR ".join('"{0}"'.format(t) for t in ts)
        rows = self.c.execute(
            "select doc_id from bm25_search('docs', ?, top_k := ?)", [expr, k]).fetchall()
        return [self.id_map[r[0]] for r in rows]

class PostgresZH:
    DSN = "host=localhost port=5432 user=bench password=" + os.environ.get("PGPASSWORD", "bench") + " dbname=postgres"
    def __init__(self, label="Postgres simple+jieba"):
        self.label = label
    def build(self, docs):
        import psycopg
        with psycopg.connect(self.DSN, autocommit=True) as c:
            c.execute("drop database if exists search_zh")
            c.execute("create database search_zh")
        self.c = psycopg.connect(self.DSN.replace("dbname=postgres", "dbname=search_zh"), autocommit=True)
        c = self.c
        c.execute("""create table docs (id text primary key, body text,
            tsv tsvector generated always as (to_tsvector('simple', coalesce(body,''))) stored)""")
        t0 = time.perf_counter()
        with c.cursor() as cur, cur.copy("copy docs (id, body) from stdin") as cp:
            for d in docs:
                cp.write_row((d["_id"], " ".join(cut_zh(d["title"] + " " + d["text"]))))
        c.execute("create index docs_tsv on docs using gin (tsv)")
        c.execute("vacuum analyze docs")
        self._build_s = time.perf_counter() - t0
        self._size = c.execute("select pg_total_relation_size('docs')").fetchone()[0]
    def search(self, q, k=100):
        toks = [t for t in cut_zh(q) if re.match(r"^[\w\u4e00-\u9fff]+$", t)]
        if not toks: return []
        # quote each lexeme; strip chars illegal in tsquery lexemes
        lex = [ "'" + t.replace("'", "") + "'" for t in toks ]
        ts = " | ".join(lex)
        sql = ("select id from docs, to_tsquery('simple', %s) query where tsv @@ query "
               "order by ts_rank(tsv, query) desc limit %s")
        return [r[0] for r in self.c.execute(sql, (ts, k)).fetchall()]

results = []
which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("all", "core", "tail"):
    if which != "tail":
        results.append(bench(SQLiteZH(W/"zh.sqlite", "trigram", pretok=False, label="SQLite FTS5 trigram (raw zh)"), "SQLite FTS5 trigram"))
        results.append(bench(SQLiteZH(W/"zh_jieba.sqlite", "unicode61", pretok=True, label="SQLite FTS5 unicode61 + jieba"), "SQLite FTS5 + jieba"))
        results.append(bench(DuckZH(W/"zh.duckdb", pretok=False, label="DuckDB FTS raw zh"), "DuckDB FTS (raw zh)"))
        results.append(bench(DuckZH(W/"zh_jieba.duckdb", pretok=True, label="DuckDB FTS + jieba (stem off)"), "DuckDB FTS + jieba"))
    results.append(bench(TantivyZH(W/"zh_jieba.tantivy", pretok=True, label="Tantivy + jieba"), "Tantivy + jieba"))
    results.append(bench(TantivyZH(W/"zh_raw.tantivy", pretok=False, label="Tantivy raw (whole-sentence token)"), "Tantivy raw zh"))
    results.append(bench(DuckTantivyZH(), "DuckDB+tantivy ext + jieba"))
    results.append(bench(PostgresZH(), "Postgres simple + jieba"))
(ROOT / f"results_zh_{which}.json").write_text(json.dumps(results, indent=1))
print("saved", f"results_zh_{which}.json")
