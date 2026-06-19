# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# 全量 ingest（默认表 chunks，TRUNCATE 后重建）
uv run scripts/ingest.py

# Contextual RAG ingest（调用 Gemini，写入 chunks_contextual 表）
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

# OCR 服务（在远端 GPU 服务器上运行，需提前安装 MinerU）
python scripts/start_ocr_service.py                    # 默认 host=127.0.0.1 port=1080
python scripts/start_ocr_service.py --host 0.0.0.0    # 对外监听

# OCR 评测（本地运行，需 SSH 隧道已建立）
uv run scripts/eval_ocr.py --api-url http://127.0.0.1:6006 --limit 50
uv run scripts/eval_ocr.py --api-url http://127.0.0.1:6006 --limit 1651          # 全量
uv run scripts/eval_ocr.py --api-url http://127.0.0.1:6006 --limit 100 --doc-types academic_literature,research_report
uv run scripts/eval_ocr.py --api-url http://127.0.0.1:6006 --limit 1651 --samples-per-type 5  # 固定测试集（每类5样本）

# OCR review（每类1样本对比 GT 与识别结果，输出 Markdown）
uv run scripts/review_ocr.py --api-url http://127.0.0.1:6006

# OCR → RAG 端到端 ingest（扫描件目录 → pgvector）
uv run scripts/ingest_ocr.py --input-dir data/scanned_docs --api-url http://127.0.0.1:6006
uv run scripts/ingest_ocr.py --input-dir data/scanned_docs --table chunks_ocr --no-truncate  # 增量追加
```

## 环境配置

`.env` 需包含（参考 `.env.example`）：
```
EMBED_API_KEY=...      # embedding 服务认证
PG_PASSWORD=...        # PostgreSQL 密码
GEMINI_API_KEY=...     # Contextual RAG（--contextual 时必须）
```

`config.yaml` 控制所有运行时参数。CLI 参数（`--overlap`、`--table`、`--reranker`）在运行时覆盖 config.yaml，不修改文件。

## 架构

### 数据流

```
CUAD (HuggingFace)
  → cuad.py            # 下载、解析，输出 QAItem + .txt 文件
  → chunking.py        # fixed / recursive 两种策略，产出 ChunkWindow
  → contextualizer.py  # （可选）调用 Gemini 为每个 chunk 生成上下文前缀
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
- **`store.py`** — 动态表名（`VectorStore(dsn, table="chunks")`），`_init_schema()` 自动建表建索引；BM25 用 PostgreSQL `tsvector`，通过 OR 语义修复了 CUAD 模板问题（见 `docs/bug_fixes.md`）；`ingest_meta` 表记录每张 chunk 表的实际 ingest 参数，`eval.py` 从此读取
- **`config.py`** — 所有 dataclass 配置，`load_config()` 从 config.yaml + .env 加载；各脚本用 `dataclasses.replace()` 在运行时覆盖字段，不改文件
- **`contextualizer.py`** — `from google import genai` 是懒加载（在 `__init__` 内），不 import 此模块不会引入 Gemini 依赖；结果缓存在 `.cache/contextual.json`，key = `chunk_id:text_hash`

### PostgreSQL 表结构

- **`chunks` / `chunks_contextual`**（或任意自定义表名）：`chunk_id PK, doc_id, text, start_pos, end_pos, embedding vector(1024), tsv tsvector GENERATED`
- **`ingest_meta`**：`table_name PK, chunk_chars, overlap, strategy, contextual, ingested_at`

### OCR 管道（独立，未接入 RAG）

```
eval_ocr.py（本地）
  → POST /file_parse（PNG 直传，MinerU 原生支持图像输入）
      → SSH 隧道（本地端口 → 远端 127.0.0.1:1080）
          → start_ocr_service.py（GPU 服务器，mineru-api FastAPI）
              → hybrid-auto-engine backend（小 VLM 辅助识别）→ Markdown
  → OmniDocBench ground truth 对比 → CER / WER
```

**OmniDocBench 数据集：** HuggingFace `opendatalab/OmniDocBench`（train split，1651张）。图像与标注分离：图像在 HF 数据集，GT 标注在独立的 `OmniDocBench.json`（首次运行自动下载到 `data/omnidocbench_annotations.json`）。

- `data_source` 字段值：`academic_literature / research_report / book / PPT2PDF / colorful_textbook / magazine / exam_paper / newspaper / note`（无 financial_report）
- GT 文本来源：`layout_dets[*].text`，过滤 `TEXT_CATS = {text_block, header, figure_caption, table_caption, page_footer, page_header}`
- 指标：CER（字符错误率）/ WER（词错误率），CER > 1.0 表示 OCR 输出远长于 GT

**OCR 服务端口：** 与 embedding 服务共用端口 1080，不同时运行。SSH 隧道本地端口 6006 → 远端 1080。

**本地依赖：** `editdistance datasets pillow httpx tqdm`（无需安装 mineru）

**OCR Baseline（hybrid-auto-engine，Run: `20260602T042922Z`，全量 1615 样本）：**

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

配置：`hybrid-auto-engine` + `parse_method=ocr` + `formula_enable=false` + `table_enable=true`；需修复 `libnvrtc-builtins.so.13.0` 软链接（实际库为 CUDA 12.8）。`research_report` 主要瓶颈为多栏布局乱序（WER 远高于 CER），非字符识别问题。

### 评估体系

**检索层（`scripts/eval.py`）：** span 匹配用 `chunk.start / chunk.end`（原始文档字符偏移），不依赖 `chunk.text` 内容。指标：hit@k、mrr@k、precision@k、recall@k。

**生成层（`scripts/eval_generation.py`）：** 三个维度：
1. **语义相似度命中率** — 生成答案与 gold answer 的 embedding cosine 相似度（阈值 0.70），比字符串包含更公平
2. **拒答准确率** — FP（无答案问题被回答）/ FN（有答案问题被拒答），基于 `generator.py` 的 JSON mode `refused` 字段
3. **LLM-as-Judge** — Faithfulness（答案忠实于上下文）/ Answer Relevancy，通过 Gemini 实现，无需 ragas 库

### 核心模块（生成层）

- **`generator.py`** — `LegalGenerator`，使用 Gemini JSON mode（`response_mime_type="application/json"`）强制结构化输出 `{"refused": bool, "answer": str}`，彻底消除软拒答歧义；`_build_context()` 注入 doc_meta 前缀
- **`contextualizer.py`** — `MetadataExtractor` 提取合同元数据（contract_type/party_a/party_b/effective_date/governing_law/key_clauses），缓存于 `.cache/meta_extract.json`
- **`store.py`** — `doc_meta` 表存储结构化元数据，`get_doc_meta(doc_id)` 供查询时注入
- **`pipeline.py`** — 新增 `get_doc_meta(doc_id)` 方法

### 当前最优配置

**检索层**（律所/法务场景，hit@1 和 mrr@5 为核心指标）：

| 参数 | 值 |
|------|----|
| table | chunks_qwen3 |
| embedding | Qwen/Qwen3-Embedding-0.6B |
| chunk_chars | 1000 / overlap=100 / strategy=recursive |
| mode | hybrid（vector + BM25 RRF 融合） |
| reranker | bge-reranker-v2-m3，top_k=60 |
| contextual | gemini-3.1-flash-lite |
| **hit@1** | **0.580** / **mrr@5=0.667** / hit@5=0.804 / hit@10=0.890 |

> 召回优先场景（尽职调查/合规审查）改用 `chunks_contextual`（BGE-M3，rerank_top_k=60）：hit@5=**0.833**

**生成层 Baseline v1**（Run: `20260527T225006Z`，200样本，30样本 RAGAS）：

| 指标 | 值 |
|------|----|
| semantic_hit_rate | 0.680（threshold=0.70） |
| false_positive_rate | 0.173 |
| false_negative_rate | 0.120 |
| faithfulness | 0.667 |
| answer_relevancy | 0.867 |
| avg_latency_ms | 803 |

配置：Gemini JSON mode + `When in doubt, refuse` + doc_meta 注入 + reranker，top_k=5，generate_k=5

**生成层 v2**（Run: `20260529T171148Z`，200样本，30样本 RAGAS）：

| 指标 | 值 |
|------|----|
| semantic_hit_rate | 0.740 |
| false_positive_rate | 0.200 |
| false_negative_rate | 0.040 |
| faithfulness | 0.500 |
| answer_relevancy | 0.967 |
| avg_latency_ms | 752 |

配置：Gemini JSON mode + `When in doubt, refuse` + few-shot 示例 + reranker，top_k=10，generate_k=8

> doc_meta 注入实验（`20260529T183159Z`）无改善，已废弃。根因：RAGAS judge 不把 doc_meta 计入上下文，模型用 meta 回答被误判为幻觉。

**生成层 v3**（Run: `20260530T110845Z`）：

| 指标 | 值 |
|------|----|
| semantic_hit_rate | 0.760 |
| false_negative_rate | 0.100 |
| faithfulness | 0.667 |
| answer_relevancy | 0.867 |

配置：逐字引用约束 + few-shot + 无 doc_meta，generate_k=8

**生成层当前最优 v4**（Run: `20260530T143927Z`，200样本，30样本 RAGAS）：

| 指标 | 值 | vs v2 | vs v3 |
|------|----|-------|-------|
| semantic_hit_rate | **0.820**（threshold=0.70） | ▲▲ +0.080 | ▲ +0.060 |
| false_positive_rate | 0.200 | = | ▼ +0.013 |
| false_negative_rate | **0.040** | = | ▲▲ -0.060 |
| faithfulness | **0.667** | ▲▲ +0.167 | = |
| answer_relevancy | **0.967** | = | ▲ +0.100 |
| avg_latency_ms | 756 | ≈ | ≈ |

配置：Gemini JSON mode + 逐字引用约束 + few-shot 示例 + **doc_meta 注入** + **RAGAS judge 包含 doc_meta 上下文** + reranker，top_k=10，generate_k=8

> **全面突破**：semantic_hit 创新高（0.820），faithfulness 维持 v3 水平（0.667），FN 恢复 v2 最低（0.040），relevancy 恢复 0.967。
> 根因：RAGAS judge 看到 doc_meta 后能正确验证元数据来源的答案，消除了测量偏差。

## 关键约束

- **API + UI 已合并为单一进程**：`serve.py` 通过 `gr.mount_gradio_app()` 在同一进程内同时提供 REST API（`/query`）和 Gradio UI（`/ui`），共享同一 `VectorStore` 连接，无锁竞争。`ui.py` 已删除。
- **切换 contextual 模式必须完整重新 ingest**（TRUNCATE + 重建），`ON CONFLICT DO NOTHING` 不会更新已有行
- Embedding 服务和 Reranker 服务共用同一个本地 endpoint（`http://127.0.0.1:6006`），需要提前启动；远程 GPU 时通过 `provider: ssh_tunnel` 配置 SSH 端口转发
- OCR 服务（MinerU）与 embedding 服务共用远端端口 1080，不同时运行；`eval_ocr.py` 所有 multipart 字段统一放入 `files=` 参数（`(None, value)` 格式），禁止同时传 `files=` 和 `data=`（httpx + h11 合并 body 时会产生 tuple 类型错误）
- Grid search 中 `data/runs/grid/20260522T*` 两次历史结果因 BM25 bug 无效，不可引用
