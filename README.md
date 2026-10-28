# fts-bench: search-without-elasticsearch, re-checked

复现并扩展 [nazmul284/search-without-elasticsearch](https://github.com/nazmul284/search-without-elasticsearch) 的嵌入式全文检索基准（BM25 / nDCG@10），新增 2026-10 时点的 DuckDB 生态选项与**中文支持对比**（首个该题面的中文基准）。

## What's here

- **Ring 1 英文复现**（SciFact 5,183 docs）：SQLite FTS5 / DuckDB FTS / Tantivy / Postgres 17，四引擎与原文逐位吻合（差 ≤0.002）
- **Ring 2 英文增量**：DuckDB FTS 去 stopwords、duckdb-tantivy 社区扩展、Tantivy raw
- **Ring 3 中文基准**（C-MTEB CmedqaRetrieval，100,001 docs / 3,999 queries，全量）：8 种配置——四引擎原生中文 vs jieba 预切词 vs trigram vs luajit 全 SQL 路线

## TL;DR

| 结论 | 数字 |
|---|---|
| 英文质量榜已被改写：原文前二 0.682/0.681 → **DuckDB FTS nostop 0.6924 / duckdb-tantivy ext 0.6839** | SciFact nDCG@10 |
| **四引擎原生中文全灭**（DuckDB 内置 FTS 把整句中文当一个 token） | nDCG 0.0005–0.004 |
| jieba 预切词后回到同档 | 0.138–0.162 |
| Postgres 中文崩溃根因 = ts_rank 无 IDF（非分词问题） | 0.0159 |
| luajit jieba 全 SQL 路线与 Python 外置管道打平 | 0.1380 vs 0.1385 |

## Results

### English (SciFact, full 300 queries)

| engine | nDCG@10 | recall@100 | p50 ms | index MB |
|---|---|---|---|---|
| DuckDB FTS nostop | **0.6924** | 0.9287 | 12.7 | 13.3 |
| duckdb-tantivy ext v0.2.0 | 0.6839 | 0.9213 | 385* | RAM |
| SQLite FTS5 porter | 0.6817 | 0.9253 | 5.1 | 17.7 |
| DuckDB FTS (paper config) | 0.6808 | 0.9260 | 8.7 | 13.3 |
| Tantivy en_stem | 0.6667 | 0.9016 | **0.31** | **2.8** |
| Postgres 17 | 0.5596 | 0.8472 | 6.6 | 19.7 |

*duckdb-tantivy ext rebuilds its in-RAM index on every call — architectural, not a bug.

### Chinese (C-MTEB CmedqaRetrieval dev, 100,001 docs / 3,999 queries)

| engine | nDCG@10 | recall@100 | p50 ms | empty |
|---|---|---|---|---|
| duckdb-tantivy ext + jieba (n=150) | **0.1622** | 0.3536 | 4300* | 0 |
| SQLite FTS5 + jieba | 0.1477 | 0.3634 | 77 | 0 |
| Tantivy + jieba | 0.1442 | 0.3522 | **1.9** | 0 |
| DuckDB FTS + jieba | 0.1385 | 0.3426 | 78 | 0 |
| **DuckDB FTS + luajit jieba (all-SQL)** | 0.1380 | 0.3231 | 78 | 0 |
| Postgres simple + jieba (n=150) | 0.0159 | 0.0751 | 348 | 0 |
| DuckDB FTS raw zh | 0.004 | 0.006 | 12 | 3451 |
| SQLite FTS5 trigram raw zh | 0.0005 | 0.0004 | 0.02 | 3995 |
| Tantivy raw zh | 0.0007 | 0.0007 | 0.02 | 3931 |

**中文三定律**：① 原生全灭（默认分词器按空格切，CJK 连续串成一个 token）；② 分词后引擎差异收窄到 2 个点内，分词模式主导质量；③ Postgres 崩溃不是分词问题——ts_rank 无 IDF 分量，高频助词淹没排序（单查询 96k 命中，目标文档排第 49,652）。

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install duckdb==1.5.6 tantivy==0.22.0 numpy psycopg pandas pyarrow jieba
# data: BEIR SciFact + mteb/CmedqaRetrieval parquet (corpus/queries/qrels) -> data/
.venv/bin/python harness/bench_local.py core   # English ring
.venv/bin/python harness/bench_local.py tune   # stopwords ablation
.venv/bin/python harness/bench_local.py ext    # duckdb-tantivy community ext
.venv/bin/python harness/bench_zh.py all       # Chinese ring (full)
.venv/bin/python harness/bench_zh_sample.py    # ext + PG on 150-query sample
.venv/bin/python harness/bench_zh_luajit.py    # all-SQL luajit jieba route
```

Requires: DuckDB ≥1.5 (community extensions: `fts`, `tantivy`, `luajit`), Postgres 17 (role `bench`), SQLite ≥3.45.

## Notes & caveats

- FiQA-2018 not re-run locally (paper numbers quoted as-is).
- ext / PG Chinese numbers are 150-query samples (seed=42); full runs infeasible (ext: 4.3 s/query × 3,999).
- luajit jieba (`duckdb-luajit-libs`) implements jieba *precise* mode; the Python ring used `cut_for_search` — that, not the engine, explains the 0.0005 gap.
- duckdb-tantivy ext usage traps found while testing: multi-word queries default to AND (need explicit OR); returns row-number BIGINT, not varchar keys (map client-side, in score order).

## Credits

- Upstream benchmark: [nazmul284/search-without-elasticsearch](https://github.com/nazmul284/search-without-elasticsearch)
- Chinese tokenizer: [fxsjy/jieba](https://github.com/fxsjy/jieba) (MIT) + its pure-Lua port in [alitrack/duckdb-luajit-libs](https://github.com/alitrack/duckdb-luajit-libs)
- Tantivy extension: [alitrack/duckdb-tantivy](https://github.com/alitrack/duckdb-tantivy)
