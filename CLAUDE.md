# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# 全量 ingest（默认表 chunks，TRUNCATE 后重建）
uv run scripts/ingest.py

# Contextual RAG ingest（调用 LLM，写入 chunks_contextual 表）
uv run scripts/ingest.py --contextual

# 指定 overlap/chunk_chars 覆盖 config.yaml
uv run scripts/ingest.py --contextual --overlap 150 --chunk-chars 1000

# 补充提取 doc_meta（不重建 chunks，只填充 doc_meta 表）
uv run scripts/ingest.py --contextual --meta-extract --table chunks_qwen3

# 检索层评估（从 DB 读取实际 ingest 参数，结果写入 data/runs/eval/<ts>.json）
uv run scripts/eval.py --reranker
uv run scripts/eval.py --reranker --table chunks_contextual

# 两个检索层结果文件 diff
uv run scripts/eval.py --compare data/runs/eval/A.json data/runs/eval/B.json

# Grid search（结果写入 data/runs/grid/<ts>/）
uv run scripts/grid_search.py --reranker

# sufficiency judge 的 A/B（判定器 vs 判定器+两段式生成，结果写入 data/runs/ab_judge/）
uv run scripts/ab_sufficiency.py --limit 200 --concurrency 4
uv run scripts/ab_sufficiency.py --compare data/runs/ab_judge/<ts>.json

# 三组 trace 语料（规格 2.6，contract scope；结果写入 data/runs/traces/）
uv run scripts/build_trace_corpus.py --limit 0 --concurrency 4
uv run scripts/build_trace_corpus.py --limit 200 --only baseline fixed   # 只跑部分组
# judge 判断 vs CUAD gold span 的逐轮 2×2 表（纯离线，不碰 DB / LLM）
uv run scripts/gold_round_check.py "data/runs/traces/<ts>_*.jsonl"

# 生成层评估（结果写入 data/runs/gen_eval/<ts>.json）
uv run scripts/eval_generation.py --limit 200 --reranker --sim-threshold 0.70
uv run scripts/eval_generation.py --limit 200 --reranker --sim-threshold 0.70 --ragas --ragas-limit 30

# Corpus 模式评估（不按 doc_id 过滤，全库检索）
uv run scripts/eval_generation.py --limit 200 --reranker --sim-threshold 0.70 --corpus

# 两个生成层结果文件 diff
uv run scripts/eval_generation.py --compare data/runs/gen_eval/A.json data/runs/gen_eval/B.json

# API + Gradio UI（http://127.0.0.1:6800/ui）
uv run scripts/serve.py
uv run scripts/serve.py --host 0.0.0.0 --port 6800
uv run scripts/serve.py --no-ui            # 仅 API，不挂载 Gradio

# 依赖安装
uv pip install -e .

# OCR 评测（走 MinerU 在线 API，需 .env 里的 MINERU_API_TOKEN）
uv run scripts/eval_ocr.py --limit 50
uv run scripts/eval_ocr.py --limit 1651                      # 全量（会超出每日高优先级额度，见下）
uv run scripts/eval_ocr.py --limit 100 --doc-types academic_literature,research_report
uv run scripts/eval_ocr.py --limit 1651 --samples-per-type 5  # 固定测试集（每类5样本）
uv run scripts/eval_ocr.py --limit 200 --model-version vlm --batch-size 100

# OCR review（每类1样本对比 GT 与识别结果，输出 Markdown）
uv run scripts/review_ocr.py

# OCR → RAG 端到端 ingest（扫描件目录 → pgvector）
uv run scripts/ingest_ocr.py --input-dir data/scanned_docs
uv run scripts/ingest_ocr.py --input-dir data/scanned_docs --table chunks_ocr --no-truncate  # 增量追加
```

## 本地运行（模型服务）

> **模型接入现状**：embedding 与 reranker 已切到 **SiliconFlow**（托管，无需本地 GPU）；
> 原先的两个本地 llama.cpp 实例（:8081 / :6006）已移除。生成 / Judge 已从 Gemini 迁到 **Z.ai GLM**。

| 用途 | 服务商 | 模型 | base_url | key |
|------|--------|------|----------|-----|
| embedding | SiliconFlow | `BAAI/bge-m3`（1024 维） | `https://api.siliconflow.cn/v1` | `EMBED_API_KEY` |
| reranker | SiliconFlow | `BAAI/bge-reranker-v2-m3` | 同上（自动拼 `/v1/rerank`） | `RERANK_API_KEY`（缺省回落 `EMBED_API_KEY`） |
| OCR | MinerU 在线 API | — | `https://mineru.net/api/v4` | `MINERU_API_TOKEN` |
| 生成 / Judge / 辅助 LLM | Z.ai | `glm-4.7-flash` | `https://api.z.ai/api/paas/v4/` | `GENERATE_MODEL_API` |

**启动顺序：先起向量库与模型服务，再跑 ingest / eval：**
```bash
# 1. 向量库（lex_rag 自己的 docker pgvector；5432 被 rag_demo、5433 被 paralegal 占用，故映射到 5434）
docker compose up -d db

# 2. Embedding / Reranker 已是托管服务（SiliconFlow），无需本地启动，只要 .env 里有 key

# 3. 换 embedding 模型后必须清旧向量缓存并重灌（否则新旧模型向量混用）
uv run scripts/ingest.py --refresh-cache        # 清 embed_cache.pkl + 重建 chunks 表
rm -f data/embed_cache_eval.pkl                 # eval 语义相似度用的缓存也要清

# 4. 评估（Langfuse Dataset + Experiment，跨 run 对比）
uv run scripts/eval_experiment.py --sync-dataset --limit 30 --run-name noreranker
uv run scripts/eval_experiment.py --limit 30 --reranker --run-name reranker

# 5. 在线问答（同时验证 Langfuse trace 树）
uv run scripts/serve.py                          # http://127.0.0.1:6800/ui
```

**关键约束：**
- **reranker 响应格式**：`direct` provider 走 `/v1/rerank`，解析 TEI 风格
  `{"results": [{"index", "score"}]}`，并兼容把分数命名为 `relevance_score` 的实现；
  `bge_http` / `macrolens` 走 `/rerank`，返回 `{"scores": [...]}`（顺序与输入一致）。
- **两个 base_url 的后缀相反**：embedding 走 OpenAI SDK，`base_url` **必须带 `/v1`**；
  reranker 自己拼 `/v1/rerank`，`base_url` **不带 `/v1`**（写了会被容错去掉）。
  embedding 与 reranker 来自同一服务商时，host 相同、后缀不同，是最容易配错的地方。
- **reranker 认证**：`RERANK_API_KEY` 未设置时回落到 `EMBED_API_KEY`（同一服务商只配一个即可）。
  key 为空则不发 `Authorization` 头，保持自建 TEI / llama.cpp 的原有行为。
- **换 embedding 模型要看维度**：`chunks.embedding` 是 `vector(1024)`，维度变了必须改表结构 + 全量重灌。
- **可观测性**：配 `.env` 的 `LANGFUSE_*` 后，在线问答自动上报 trace 树、`eval_experiment` 上报 Experiment scores；未配则完全 no-op（见 `lex_rag/tracing.py`）。

## 环境配置

`.env` 需包含（参考 `.env.example`）：
```
EMBED_API_KEY=...      # embedding 服务认证
PG_PASSWORD=...        # PostgreSQL 密码
MINERU_API_TOKEN=...   # MinerU 在线 API（OCR 相关脚本必须）
GENERATE_MODEL_API=... # 生成 / Judge / Contextual RAG / HyDE / multi-query
LANGFUSE_PUBLIC_KEY=…  # 可选：LLM 可观测性（留空则 no-op）
LANGFUSE_SECRET_KEY=…
LANGFUSE_HOST=https://cloud.langfuse.com
```

`config.yaml` 控制所有运行时参数。CLI 参数（`--overlap`、`--table`、`--reranker`）在运行时覆盖 config.yaml，不修改文件。

## 架构

### 数据流

```
CUAD (HuggingFace)
  → cuad.py            # 下载、解析，输出 QAItem + .txt 文件
  → chunking.py        # fixed / recursive 两种策略，产出 ChunkWindow
  → contextualizer.py  # （可选）调用 LLM 为每个 chunk 生成上下文前缀
  → embeddings.py      # OpenAI-compatible API（BAAI/bge-m3），带 pickle 缓存
  → store.py           # PostgreSQL + pgvector，写入 chunks 表
```

查询时：
```
question → embeddings.py → store.py（vector / bm25 / hybrid RRF）
  → reranker.py（可选，TEI /v1/rerank）→ 返回 ChunkWindow 列表
```

### 核心模块

- **`pipeline.py`** — 唯一的对外入口，封装 ingest + query 两条路径，组合上面所有模块
- **`store.py`** — 所有游标走 `_cursor()`，**出错必回滚**：psycopg 在事务中出错后该连接上的后续语句一律报 `InFailedSqlTransaction` 直到显式 rollback，不回滚会让一次瞬时错误**永久**报废连接。`serve.py` 是长驻进程，症状会是"从此所有查询都失败"而不是"那一次失败"（实测 1 次死锁连锁出 15 条 InFailedSqlTransaction）。并发构造 `VectorStore` 会因 `_init_schema()` 的 DDL 互相死锁——多线程时要**串行**建好连接再起线程，或传 `init_schema=False`。动态表名（`VectorStore(dsn, table="chunks")`），`_init_schema()` 自动建表建索引；BM25 用 PostgreSQL `tsvector`，通过 OR 语义修复了 CUAD 模板问题（见 `docs/bug_fixes.md`）；`ingest_meta` 表记录每张 chunk 表的实际 ingest 参数，`eval.py` 从此读取
- **`config.py`** — 所有 dataclass 配置，`load_config()` 从 config.yaml + .env 加载；各脚本用 `dataclasses.replace()` 在运行时覆盖字段，不改文件
- **`llm.py`** — `ChatClient`，所有 LLM 调用的唯一入口（OpenAI 兼容 `/chat/completions`）。换服务商只改 `config.yaml` 的 `contextual.base_url` / `contextual.model`，调用方不动
- **`contextualizer.py`** — 五个类都用 `ChatClient.from_config(cfg, max_retries=0)`：各类自带重试循环，不覆盖会变成 (n+1)² 次请求；结果缓存在 `.cache/contextual.json`，key = `chunk_id:text_hash`
  ⚠️ **降级结果绝不写缓存**（HyDE / MetadataExtractor / QueryExpander 三处）。
  写了就把一次瞬时故障固化成永久行为，而且完全静默：`.cache/hyde.json` 曾 41 条全部
  等于原问题，HyDE 因此长期是空操作，唯一症状是 `hybrid vs hyde` 检索重合度精确等于
  1.000。判据必须是"这次有没有降级"，不能是"结果是否等于输入"——模型合法的短回答会
  被后者误判。见 `docs/experiments.md`
- **`agent.py`** — `AgenticPipeline`，规格 2.4 的多轮循环：选策略 → 检索 → 累积 →
  `SufficiencyJudge` 判定够不够。继续条件是"不充分"而非旧版的"结果为空"（hybrid 几乎
  永不返回空，旧循环是死代码）。`StrategySelector` 是 LLM 决策，失败/重复时回落到
  `missing_kind` 映射表。**防重复在执行层拦截**，不靠 prompt。传 `sink=TraceSink(...)`
  即逐轮落盘。`query()` / `query_stream()` 的签名保持不变，`serve.py` 与 `eval.py` 不用改

### 检索分数

`ChunkWindow.score` / `score_kind` 在检索期填充，ingest 期恒为 None。**一律"越大越相关"**
（余弦距离在 store 里已转成相似度），但**数值跨阶段不可比**，所以 `score_kind` 必须跟着
分数一起落盘：

| score_kind | 来源 | 典型量级 |
|-----------|------|---------|
| `cosine_sim` | `store.search_vector`，1 − 余弦距离 | 0.5 ~ 0.7 |
| `bm25` | `store.search_bm25`，`ts_rank_cd` | 0.8 ~ 1.5 |
| `rrf` | `store.search_hybrid` / `pipeline._rrf_merge` | 0.01 ~ 0.05 |
| `rerank` | `reranker.rerank` | 0 ~ 1 |

后一阶段会覆盖前一阶段的分数：返回的是哪个阶段的排名，分数就是哪个阶段的，否则 trace 里
的分数与它自己的名次对不上。`expand_to_parent` 的 parent 继承命中它的 child 里最高的分数。

> ⚠️ 改检索 SQL 或排序逻辑后，**先对拍返回的 chunk_id 顺序，再考虑跑评测**：
> 200 条对拍 2 分钟且能定位到"哪一条第几位"，跑全量要 27 分钟且只能给出聚合差值。
> 见 `docs/refactor_regression.md`。

### PostgreSQL 表结构

- **`chunks` / `chunks_contextual`**（或任意自定义表名）：`chunk_id PK, doc_id, text, start_pos, end_pos, embedding vector(1024), tsv tsvector GENERATED`
- **`ingest_meta`**：`table_name PK, chunk_chars, overlap, strategy, contextual, ingested_at`

### OCR 管道（独立，未接入 RAG）

```
eval_ocr.py（本地，按 --batch-size 成批）
  → lex_rag/ocr.py  MinerUCloudClient
      ① POST /api/v4/file-urls/batch   申请预签名上传链接（batch_id + file_urls）
      ② PUT  file_urls[i]              PNG 直传 OSS（不带 Authorization、不设 Content-Type）
      ③ GET  /api/v4/extract-results/batch/{batch_id}   轮询到 state=done
      ④ GET  full_zip_url              下载 zip，取出 full.md
  → OmniDocBench ground truth 对比 → CER / WER
```

**OmniDocBench 数据集：** HuggingFace `opendatalab/OmniDocBench`（train split，1651张）。图像与标注分离：图像在 HF 数据集，GT 标注在独立的 `OmniDocBench.json`（首次运行自动下载到 `data/omnidocbench_annotations.json`）。

- `data_source` 字段值：`academic_literature / research_report / book / PPT2PDF / colorful_textbook / magazine / exam_paper / newspaper / note`（无 financial_report）
- GT 文本来源：`layout_dets[*].text`，过滤 `TEXT_CATS = {text_block, header, figure_caption, table_caption, page_footer, page_header}`
- 指标：CER（字符错误率）/ WER（词错误率），CER > 1.0 表示 OCR 输出远长于 GT

**MinerU 在线 API 约束（截至 2026-08）：**
- 免费额度：每账号每天 1000 页最高优先级（另一处文档写 2000），**超出只降优先级、不拒绝**——
  全量 1651 张一次跑完会有后半程明显变慢，做延迟对比时要用小测试集。
- 单文件 ≤200MB / ≤200 页；单批 ≤200 个文件；上传链接有效期 24 小时。
- 官方定位是"内测 + 免费试用"，没有 SLA、没有公开价目表，商用授权需单独确认。

**本地依赖：** `editdistance datasets pillow httpx tqdm`（无需安装 mineru）

**OCR Baseline（⚠️ 本地自建服务时期的历史数字，Run: `20260602T042922Z`，全量 1615 样本）：**

> 线上 API 没有 `backend` / `parse_method` 参数（只有 `model_version` + `is_ocr`），
> 下表与线上结果**不可直接比较**，切换后需要重跑基线。

| 类型 | CER | WER | vs pipeline |
|------|-----|-----|-------------|
| research_report | 27.24% | 39.96% | CER ▼4.88 |
| book | 13.51% | 15.77% | CER ▼6.53 |
| academic_literature | 11.02% | 12.24% | CER ▼3.49 |
| note | 2.96% | 2.99% | CER ▼5.17 |
| magazine | 3.10% | 3.58% | CER ▼1.31 |
| colorful_textbook | 3.00% | 2.76% | CER ▼1.16 |
| ppt2pdf | 3.43% | 4.61% | CER ▼0.76 |
| exam_paper | 0.80% | 1.11% | CER ▼0.39 |
| newspaper | 0.29% | 0.88% | CER ▼0.39 |
| **Overall** | **7.35%** | **9.22%** | **CER ▼2.75** |

配置：`hybrid-auto-engine` + `parse_method=ocr` + `formula_enable=false` + `table_enable=true`（自建 mineru-api，已下线）。`research_report` 主要瓶颈为多栏布局乱序（WER 远高于 CER），非字符识别问题。

### 评估体系

**检索层（`scripts/eval.py`）：** span 匹配用 `chunk.start / chunk.end`（原始文档字符偏移），不依赖 `chunk.text` 内容。指标：hit@k、mrr@k、precision@k、recall@k。

**生成层（`scripts/eval_generation.py`）：** 三个维度：
1. **语义相似度命中率** — 生成答案与 gold answer 的 embedding cosine 相似度（阈值 0.70），比字符串包含更公平
2. **拒答准确率** — FP（无答案问题被回答）/ FN（有答案问题被拒答），基于 `generator.py` 的 JSON mode `refused` 字段
3. **LLM-as-Judge** — Faithfulness（答案忠实于上下文）/ Answer Relevancy，通过 `ChatClient` 实现，无需 ragas 库

### 核心模块（生成层）

- **`generator.py`** — `LegalGenerator`，用 `response_format={"type":"json_object"}` 输出 `{"refused": bool, "answer": str}`，消除软拒答歧义；`_build_context()` 注入 doc_meta 前缀。
  ⚠️ 与 Gemini 时期的差异：OpenAI 风格没有服务端 `response_schema`，**只保证语法合法、不保证字段齐全**，结构约束退化为 prompt 约束，`_parse_response` 的字段缺省逻辑成为最后一道防线
- **`contextualizer.py`** — `MetadataExtractor` 提取合同元数据（contract_type/party_a/party_b/effective_date/governing_law/key_clauses），缓存于 `.cache/meta_extract.json`
- **`store.py`** — `doc_meta` 表存储结构化元数据，`get_doc_meta(doc_id)` 供查询时注入
- **`pipeline.py`** — 新增 `get_doc_meta(doc_id)` 方法
- **`trace_sink.py`** — 本地 JSONL 实验语料落盘（规格 2.5），**与 `tracing.py` 职责不同**：
  后者是 Langfuse 封装、没配 key 就完全 no-op（在线可观测性）；前者给了路径就一定写，
  写不出来往 stderr 响一次但不抛（实验语料）。一次查询一行，写完立刻 `flush` + `fsync`
  ——此前两轮评测在收尾阶段崩掉、200 条结果全丢，逐行落盘是针对那个教训的。
  路径以 `.gz` 结尾自动压缩。读回用 `read_traces()` / `read_meta()`
- **`sufficiency.py`** — `SufficiencyJudge`，判断"当前 chunks 够不够回答"，输出
  `sufficient / missing / missing_kind / out_of_scope / confidence`。
  `missing_kind` 是**枚举**（exact_term / clause_context / concept_mismatch /
  multi_aspect），经 `STRATEGY_HINT` 映射到下一轮该换的检索策略——读者是代码不是人，
  自由文本会把决策质量押在措辞上。字段缺失时**一律按"不够"缺省**：判成不够最多白烧
  一轮，判成够了会拿着残缺上下文直接生成。
  ⚠️ `confidence` 在两个分支上重叠（0.95~1.00 vs 0.80~1.00），**不要拿它设阈值**；
  ⚠️ `mode="unified"` 与 `generator.VerifiedGenerator` 是输掉 A/B 的那一臂，只为复跑
  实验保留，不要接进生产路径

### 当前最优配置

> ⚠️ **按 `doc_id` 检索时策略选择没有发挥空间。** CUAD 的 25 份合同中位只有 23 个
> chunk，17/25 的 chunk 总数 ≤ `fetch_k=60`——候选池就是整份合同，换策略不改变
> reranker 的输入。实测 hybrid / vector / hyde 三者返回 top-10 的重合度 0.97~0.99，
> 五个动作实际塌缩成两个（hybrid 系 与 bm25）；corpus scope 下才拉得开
> （bm25 vs vector 降到 0.333）。
>
> 但**不要因此换到 corpus scope**：那边 CUAD 的问题不标识文档，任务不可解。
> 1000 条样本只有 41 个不同的问题文本，每条原样出现在 24~25 份合同里，查询里没有
> 信息能指出该找哪一份。实测 20 条 pilot：corpus 下 gold span 只在 1 轮里出现过，
> contract 下是 16 轮。**评测与语料产出一律用 contract scope**，`--corpus` 只用于
> 复现这个对照。策略空间小是要如实记录的性质，不是换 scope 的理由——换到一个自变量
> 能动但因变量无意义的设定更糟。见 `docs/experiments.md`。

**检索层**（律所/法务场景，hit@1 和 mrr@5 为核心指标）。
Run: `20260824T230237Z`，1000 条 CUAD QA：

| 参数 | 值 |
|------|----|
| table | chunks |
| embedding | SiliconFlow `BAAI/bge-m3`（1024 维） |
| chunk_chars | 1000 / overlap=100 / strategy=recursive |
| mode | hybrid（vector + BM25 RRF 融合） |
| reranker | SiliconFlow `BAAI/bge-reranker-v2-m3`，rerank_top_k=60，batch_size=60 |
| contextual | 未开启 |
| **hit@1** | **0.541** / **mrr@5=0.640** / hit@5=0.815 / hit@10=0.865 / recall@5=0.748 |

> **与迁移前同配置对比无退化**：`docs/baseline.md` 阶段四"无 Contextual + overlap=100"
> 一行（自建 BGE-M3）是 hit@1=0.516 / hit@10=0.843 / mrr@5=0.631，本次分别为
> 0.541 / 0.865 / 0.640。同一个 bge-m3 换了托管方，向量质量一致。
>
> 历史最优 hit@1=0.580 是 `chunks_qwen3` + **开启 contextual** 的配置，与上表不是同一组
> 参数，**不可直接比较**。要复现需重跑一次 contextual ingest（每 chunk 调一次 LLM）。

> 生成层 v1–v3 的历史指标与配置见 `docs/experiments.md`。

**生成层当前最优 v5**（Run: `20260826T*`，200 样本，30 样本 judge，`errors=0`，`n_judge_failed=0`）：

| 指标 | v5 (qwen3.7-flash) | v4 (Gemini) | 说明 |
|------|-------------------|-------------|------|
| **false_positive_rate** | **0.120** | 0.200 | ✅ 编造答案少近一半 |
| false_negative_rate | 0.060 | 0.040 | 基本追平 |
| semantic_hit_rate | 0.800 | 0.820 | 基本追平 |
| 判别力 J | **0.820** | — | 正确拒答率 − 误拒率 |
| answer_relevancy | 0.857 | 0.967 | ⚠️ judge 不同，不可比 |
| faithfulness | 0.500 | 0.667 | ⚠️ judge 不同，不可比 |
| avg_latency_ms | 7774 | 756 | ❌ thinking 的代价 |

配置：qwen3.7-flash + `thinking=true` + `json_object` + KIND A/B 分流 prompt + 逐字引用约束
+ few-shot + reranker，top_k=10，generate_k=8；judge = qwen3.7-plus（不同家族，避免自评偏袒）

> **judge 换了模型，faithfulness / answer_relevancy 与 v4 不可比**——那是两把尺子。
> FP / FN / semantic_hit 不依赖 judge，只有这三行的对比成立。
> 选定 judge 后应冻结不动，否则每换一次就切断一次历史可比性。
>
> 剩余差距是延迟（7.8s vs 756ms），全部来自 thinking。
>
> ⚠️ **"关掉 thinking 会让 J 从 0.60 掉到 0.40"这个说法只在 10 条横评上成立。**
> 2026-08-26 的 200 条实测里，关掉 thinking 的单段路径 J=0.833、开着的 A 臂
> J=0.820，差异在噪声内（|z|<1），而延迟是 3.4s vs 10.9s。也就是说 thinking 花掉
> 3 倍延迟买到的质量差异测不出来。要动配置需要一次单变量实验（同模型同 prompt、
> 只切 thinking、200 条），**尚未做，所以配置未改**。见 `docs/experiments.md`。
>
> 曾经的候选优化"两段式分流"（快速生成 + 廉价校验）已被 200 条 A/B 否掉：校验环节
> 只干预 3 次且全错，成本翻倍而每个指标都变差。原因是第二段与第一段同模型同上下文，
> 不掌握新信息。
>
> 迁移过程中的失败定位（GLM 拒答门塌陷、thinking 的双向效应、json_schema 无效）
> 见 `docs/experiments.md`。

### Agentic 循环的实测结论（2026-08-27，1000 条 × 3 组，contract scope）

`data/runs/traces/20260827T015344Z_*.jsonl`。三组对照的结果是**负结果**，写在这里
免得后来者重跑一遍：

| 组 | accuracy | 白烧率 | 提前停止率 | 平均轮数 | J |
|----|----------|--------|-----------|---------|---|
| baseline（LLM 选策略） | 0.469 | 0.624 | 0.026 | 1.46 | 0.668 |
| **fixed（固定阶梯，无 LLM 决策）** | 0.478 | 0.623 | 0.037 | 1.43 | 0.660 |
| no_rerank | 0.473 | 0.663 | 0.051 | 1.45 | 0.663 |

- **LLM 策略选择器无可测收益**：与"不让系统自己选"的 fixed 组差 0.009。选择器确实
  在做不同选择（bm25 264 / hyde 169 / 重选 21），只是选什么都一样。
  ⚠️ 混淆因素：contract scope 下候选池就是整份合同，五个动作塌缩成两个——"选择器
  没用"与"这个 scope 下没得选"在本轮数据里分不开。
- **多轮成本收益比极差**：281 条有答案样本，多跑 218 轮只换回 7 条"第 0 轮没拿到、
  后来拿到了"（约 31 轮救回 1 条）。检索在第一轮就见顶。
- **判定器极度保守**：白烧 0.624 / 提前停止 0.026，与 #24 在 200 条上的形态一致。
  方向是安全的那一侧（提前停止直接产出错答案），62% 的白烧率是当时最该动的地方。
  **已改，见下。**

详见 `docs/experiments.md`。

### 判定器改 KIND A/B 分流后的现状（2026-08-27，语料 `20260827T111059Z_baseline.jsonl`）

判定器的 prompt 原本要求"回答问题的那段文字必须字面在场"，于是它去找一个**标着**
该概念的条款——合同标题在上下文里却报"没有 document name"，签名块在上下文里却报
"没有明确定义 Parties"。事实抽取类问题白烧率 0.75、条款存在性类 0.41，差距全在这。
生成层 v5 早就用 KIND A / KIND B 分流修过同一个病，判定器当时没拿到。

| | 旧 | 新 |
|---|---|---|
| accuracy | 0.469 | **0.637** |
| 白烧率 | 0.624 | **0.425** |
| 提前停止率 | 0.026 | 0.068 |
| out_of_scope 的 J | 0.677 | 0.639 |
| 平均轮数 | 1.46 | 1.51 |

⚠️ **降白烧不等于降成本**：平均轮数反而从 1.46 升到 1.51。省下的 114 个白烧轮次
落在有答案样本上，被无答案样本上少触发的 21 次 `out_of_scope` 提前退出抵消了。
买到的是判断质量，不是钱。

⚠️ **放松"充分"的判据会连带放松 `out_of_scope`**。中间那版加了一条 "For both
kinds" 的通则，结果 205 条丢掉的正确拒答里 189 条来自 KIND A——判定器把"再检索也
没用"读成了 `sufficient` 而不是 `out_of_scope`，J 从 0.677 塌到 0.460。改判定器
prompt 时**必须同时看 2×2 表和 J**，只看 2×2 会漏掉全部 719 条无答案样本。

> `terminated_by=refused` **不是系统拒答**，只是循环提前退出；chunks 照样返回，
> 拒答由生成层自己的门决定（`eval.py` 只把它当日志字段）。所以上表的 J 衡量的是
> 判定器 `out_of_scope` 信号的质量，不是端到端拒答率。

## 关键约束

- **API + UI 已合并为单一进程**：`serve.py` 通过 `gr.mount_gradio_app()` 在同一进程内同时提供 REST API（`/query`）和 Gradio UI（`/ui`），共享同一 `VectorStore` 连接，无锁竞争。`ui.py` 已删除。
- **切换 contextual 模式必须完整重新 ingest**（TRUNCATE + 重建），`ON CONFLICT DO NOTHING` 不会更新已有行
- Embedding / Reranker endpoint 由 `config.yaml` 的 `embedding.base_url` / `reranker.base_url` 指定，需要提前启动；两者可以是同一个服务，也可以分开。远程 GPU 时通过 `provider: ssh_tunnel` 配置 SSH 端口转发
- OCR 走 MinerU 在线 API（`lex_rag/ocr.py`），自建 mineru-api 服务与 SSH 隧道已移除。上传到预签名 URL 时**不要设置 Content-Type、不要带 Authorization**（签名已在 URL 里，多带会 403）；轮询结果必须按 `data_id` 回填而不是按返回顺序——顺序错位不会报错，只会让每个样本对上别人的 ground truth
- Grid search 中 `data/runs/grid/20260522T*` 两次历史结果因 BM25 bug 无效，不可引用
