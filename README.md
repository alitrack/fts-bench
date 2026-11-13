# fts-bench: search-without-elasticsearch, re-checked

复现并扩展 [nazmul284/search-without-elasticsearch](https://github.com/nazmul284/search-without-elasticsearch) 的嵌入式全文检索基准（BM25 / nDCG@10），新增 2026-10 时点的 DuckDB 生态选项与**中文支持对比**（首个该题面的中文基准），并在 2026-11 复测 [duckdb-tantivy](https://github.com/alitrack/duckdb-tantivy) v0.4.0（原生 jieba + 持久索引）。

## What's here

- **Ring 1 英文复现**（SciFact 5,183 docs）：SQLite FTS5 / DuckDB FTS / Tantivy / Postgres 17，四引擎与原文逐位吻合（差 ≤0.002）
- **Ring 2 英文增量**：DuckDB FTS 去 stopwords、duckdb-tantivy 社区扩展、Tantivy raw
- **Ring 3 中文基准**（C-MTEB CmedqaRetrieval，100,001 docs / 3,999 queries，全量）：8 种配置——四引擎原生中文 vs jieba 预切词 vs trigram vs luajit 全 SQL 路线
- **Ring 4 v0.4.0 复测**（2026-11-13）：duckdb-tantivy 持久命名索引 + 扩展内建 jieba——中文 ext 配置首次跑通**全量 3,999 查询**（10 月只能 150 样本）

## TL;DR

| 结论 | 数字 |
|---|---|
| 英文质量榜已被改写：原文前二 0.682/0.681 → **DuckDB FTS nostop 0.6924 / duckdb-tantivy ext 0.6839** | SciFact nDCG@10 |
| **持久索引让 ext 延迟降 265×**（385 ms → 1.45 ms p50），nDCG 逐位不变（0.6839） | SciFact, v0.2.0 vs v0.4.0 |
| ext 内建 jieba（v0.2.1+）：原生 `language := 'chinese'` 无需外部预切词 | 中文全量 nDCG 0.1409 |
| ext 中文首次全量跑通（10 月 150 样本 4.3 s/query → 全量 26 ms/query） | 3,999 queries |
| **四引擎原生中文全灭**（DuckDB 内置 FTS 把整句中文当一个 token） | nDCG 0.0005–0.004 |
| jieba 分词后各引擎回到同档 | 0.138–0.162 |
| Postgres 中文崩溃根因 = ts_rank 无 IDF（非分词问题） | 0.0159 |
| luajit jieba 全 SQL 路线与 Python 外置管道打平 | 0.1380 vs 0.1385 |

## Results

### English (SciFact, full 300 queries)

| engine | nDCG@10 | recall@100 | p50 ms | index MB |
|---|---|---|---|---|
| DuckDB FTS nostop | **0.6924** | 0.9287 | 12.7 | 13.3 |
| duckdb-tantivy v0.4.0 persistent | 0.6839 | 0.9213 | **1.45** | 3.0 |
| duckdb-tantivy ext v0.2.0 (ephemeral) | 0.6839 | 0.9213 | 385* | RAM |
| SQLite FTS5 porter | 0.6817 | 0.9253 | 5.1 | 17.7 |
| DuckDB FTS (paper config) | 0.6808 | 0.9260 | 8.7 | 13.3 |
| Tantivy en_stem | 0.6667 | 0.9016 | 0.31 | 2.8 |
| Postgres 17 | 0.5596 | 0.8472 | 6.6 | 19.7 |

*v0.2.0 `bm25_search` rebuilds its in-RAM index on every call — architectural, not a bug. v0.3.0+ persistent named indexes fix exactly this: build once (0.4 s), search at 1.45 ms p50. Persistent vs ephemeral results verified **bit-identical over all 300 queries** (0 diffs).

### Chinese (C-MTEB CmedqaRetrieval dev, 100,001 docs / 3,999 queries)

| engine | nDCG@10 | recall@100 | p50 ms | empty |
|---|---|---|---|---|
| duckdb-tantivy ext + jieba (n=150, v0.2.0 ephemeral) | **0.1622** | 0.3536 | 4300* | 0 |
| SQLite FTS5 + jieba | 0.1477 | 0.3634 | 77 | 0 |
| duckdb-tantivy v0.4.0 ephemeral native-jieba (n=150) | 0.1480 | 0.3314 | 8570 | 0 |
| Tantivy + jieba | 0.1442 | 0.3522 | **1.9** | 0 |
| duckdb-tantivy v0.4.0 persistent zh native-jieba (full 3,999) | 0.1409 | 0.3357 | 26 | 0 |
| DuckDB FTS + jieba | 0.1385 | 0.3426 | 78 | 0 |
| DuckDB FTS + luajit jieba (all-SQL) | 0.1380 | 0.3231 | 78 | 0 |
| Postgres simple + jieba (n=150) | 0.0159 | 0.0751 | 348 | 0 |
| DuckDB FTS raw zh | 0.004 | 0.006 | 12 | 3451 |
| Tantivy raw zh | 0.0007 | 0.0007 | 0.02 | 3931 |
| duckdb-tantivy v0.4.0 zh AND-mode (full 3,999) | 0.0007 | 0.0006 | 0.89 | 3994 |
| SQLite FTS5 trigram raw zh | 0.0005 | 0.0004 | 0.02 | 3995 |

**中文三定律**：① 原生全灭（默认分词器按空格切，CJK 连续串成一个 token）；② 分词后引擎差异收窄到 2 个点内，分词模式主导质量；③ Postgres 崩溃不是分词问题——ts_rank 无 IDF 分量，高频助词淹没排序（单查询 96k 命中，目标文档排第 49,652）。

**v0.4.0 附注（2026-11-13 复测）**：
- ext 内建 jieba（`language := 'chinese'`）后中文索引 9.8 s 建成（100,001 docs / 54.5 MB），全量 3,999 查询 26 ms p50——10 月的 ephemeral 配置是 4.3 s/query，全量不可行；持久索引把「不可行」变「常规」。
- 内建 jieba 用的是 jieba *precise* 模式：同 150 样本上 v0.2.0（外部 `cut_for_search` 预切词索引）0.1622 vs v0.4.0（内建 precise 索引，外部 `cut_for_search` 查询）0.1480——差距来自索引侧分词模式，不是引擎；查询侧统一 `cut_for_search` OR 连接。
- AND 默认语义对中文问答查询几乎全灭（3,994/3,999 空结果）——多词查询必须显式 OR，与英文环结论一致。
- ephemeral 中文（v0.4.0, n=150）p50 8.6 s/query，比 v0.2.0 的 4.3 s 更慢——内建 jieba 的切词开销叠加每查询重建，进一步坐实「重复搜索必须用持久索引」。

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install duckdb==1.5.6 tantivy==0.22.0 numpy psycopg pandas pyarrow jieba
.venv/bin/python harness/get_data.py            # SciFact + CmedqaRetrieval (HF mirrors, ~65 MB)
# duckdb-tantivy v0.4.0 artifact (unsigned; community registry still serves v0.2.0 for 1.5.6):
curl -L -o /tmp/tantivy.duckdb_extension \
  https://github.com/alitrack/duckdb-tantivy/releases/download/v0.4.0/tantivy-linux_amd64.duckdb_extension
.venv/bin/python harness/bench_local.py core    # English ring
.venv/bin/python harness/bench_local.py tune    # stopwords ablation
.venv/bin/python harness/bench_local.py ext     # duckdb-tantivy community ext (v0.2.0 via registry)
.venv/bin/python harness/bench_zh.py all        # Chinese ring (full)
.venv/bin/python harness/bench_zh_sample.py     # ext + PG on 150-query sample
.venv/bin/python harness/bench_zh_luajit.py     # all-SQL luajit jieba route
.venv/bin/python harness/bench_v040.py en       # v0.4.0 persistent + bit-identity (English)
.venv/bin/python harness/bench_v040.py zh       # v0.4.0 native-jieba, FULL 3,999 queries
```

Requires: DuckDB ≥1.5 (community extensions: `fts`, `tantivy`, `luajit`), Postgres 17 (role `bench`), SQLite ≥3.45.

## Notes & caveats

- FiQA-2018 not re-run locally (paper numbers quoted as-is).
- ext / PG Chinese numbers are 150-query samples (seed=42); full runs infeasible (ext: 4.3 s/query × 3,999).
- v0.4.0 numbers (Ring 4) load the unsigned GitHub-release artifact; the community registry still serves v0.2.0 for duckdb 1.5.6 (update in review).
- beir-cellar/beir GitHub no longer hosts dataset files (404) — `harness/get_data.py` uses HF mirrors (`BeIR/scifact`, `BeIR/scifact-qrels`, `C-MTEB/CmedqaRetrieval`, `C-MTEB/CmedqaRetrieval-qrels`).
- luajit jieba (`duckdb-luajit-libs`) implements jieba *precise* mode; the Python ring used `cut_for_search` — that, not the engine, explains the 0.0005 gap.
- duckdb-tantivy ext usage traps found while testing: multi-word queries default to AND (need explicit OR); returns row-number BIGINT, not varchar keys (map client-side, in score order).

## Credits

- Upstream benchmark: [nazmul284/search-without-elasticsearch](https://github.com/nazmul284/search-without-elasticsearch)
- Chinese tokenizer: [fxsjy/jieba](https://github.com/fxsjy/jieba) (MIT) + its pure-Lua port in [alitrack/duckdb-luajit-libs](https://github.com/alitrack/duckdb-luajit-libs)
- Tantivy extension: [alitrack/duckdb-tantivy](https://github.com/alitrack/duckdb-tantivy)
