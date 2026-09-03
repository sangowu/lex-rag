# ⚖️ Legal RAG — Contract Question-Answering with Citations

A production-style **Retrieval-Augmented Generation** system for legal contracts. Ask a question about a contract and get a grounded answer with **verbatim citations** — or an explicit refusal when the contract has no answer.

Built end-to-end on the [CUAD](https://www.atticusprojectai.org/cuad) contract-understanding dataset: hybrid retrieval, cross-encoder reranking, structured-output generation with a hard refusal gate, and a **full evaluation harness** that tracks both the retrieval layer and the generation layer.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white">
  <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white">
  <img alt="Qwen" src="https://img.shields.io/badge/LLM-Qwen3.7--Flash-4F46E5">
  <img alt="uv" src="https://img.shields.io/badge/packaging-uv-DE5FE9">
</p>

---

## Demo

Two questions against the **same** contract (`LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT`), copied verbatim out of evaluation run [`20260829T111207Z`](data/runs/gen_eval/20260829T111207Z.json) — these are recorded system outputs, not hand-written illustrations.

**① The contract answers it → verbatim quote with a citation**

> **Q** — *Highlight the parts (if any) of this contract related to "Renewal Term"… What is the renewal term after the initial term expires?*
>
> **A** — "If Distributor complies with all of the terms of this Agreement, the Agreement shall be renewable on an annual basis for one (1) year terms for up to another ten (10) years on the same terms and conditions as set forth herein." **[2]**
>
> `refused: false` · cosine vs. gold **0.97** · 1 800 ms · 6 954 prompt + 65 completion tokens

**② The contract is silent → explicit refusal, nothing invented**

> **Q** — *…related to "Most Favored Nation"… Is there a clause that if a third party gets better terms on the licensing or sale of technology…?*
>
> **A** — *(empty)*
>
> `refused: true` · 1 070 ms · 15 completion tokens

The second case is the one that matters here. The model emits `{"refused": true}` as a **structured field** rather than hedging in prose, so "didn't answer" and "answered wrong" stay separable all the way into the metrics. Over 200 questions (50 answerable, 150 not): **false-answer 0.060** — 9 of the 150 unanswerable questions still got an answer, which is the remaining hallucination surface — and **false-refusal 0.120**.

> _Gradio UI screenshot still to be added; `/ui` renders these same two paths._

---

## Why this project is interesting

Most RAG demos stop at "embed → retrieve → prompt". This one is built like an engineering deliverable:

- **Hybrid retrieval** — dense vectors **+** PostgreSQL BM25, fused with Reciprocal Rank Fusion (RRF), then re-ranked by a cross-encoder.
- **Hallucination control** — the generator runs in **JSON mode** and must emit `{"refused": bool, "answer": str}`. Refusal is a first-class signal, not a string match — critical for a legal use case where a confident wrong answer is worse than "I don't know".
- **A real evaluation harness** — separate metrics for retrieval (`hit@k`, `mrr@k`, `precision/recall`) and generation (semantic-hit rate, false-refusal / false-answer rates, and LLM-as-judge faithfulness & answer-relevancy). Every config change is tracked as a versioned run with before/after numbers.
- **An agentic retrieval loop** — an LLM **sufficiency judge** decides whether the retrieved chunks can actually answer the question, and if not, which strategy to try next. It is measured, not assumed: on 1 000 questions it scores accuracy 0.759 against the CUAD gold spans.
- **An advanced-technique playground** — Contextual RAG, HyDE, multi-query expansion and parent-child chunking are each individually toggleable, so each can be A/B'd against the baseline. **Negative results are kept** — the LLM strategy selector, two-stage verified generation and two attempts at a judge-prompt rewrite all failed their A/Bs and are written up in `docs/experiments.md` rather than quietly dropped.
- **An OCR pipeline wired into ingest** — scanned contracts → Markdown via MinerU → the same pgvector store (`scripts/ingest_ocr.py`), benchmarked on OmniDocBench along the way.

---

## Architecture

```mermaid
flowchart LR
    subgraph Ingest
        A[CUAD contracts] --> B[Chunking<br/>recursive / parent-child]
        B --> C{Contextual RAG?<br/>optional LLM prefix}
        C --> D[Embeddings<br/>BGE-M3]
        D --> E[(PostgreSQL + pgvector<br/>vector + tsvector/BM25)]
    end

    subgraph Query
        Q[Question] --> R{HyDE / Multi-query?<br/>optional}
        R --> S[Hybrid search<br/>vector + BM25 → RRF]
        S --> T[Cross-encoder rerank<br/>bge-reranker-v2-m3]
        T --> U[Generator<br/>JSON mode + refusal gate]
        U --> V[Answer + verbatim citations]
    end

    E --- S
```

**Serving:** a single process exposes both a **REST API** (`POST /query`, with SSE streaming and an optional agentic mode) and a **Gradio chat UI** (`/ui`), mounted together via `gr.mount_gradio_app()` so they share one `VectorStore` connection.

---

## Results

> Dataset: CUAD. Embedding dim 1024. All numbers are reproducible via the `scripts/eval*.py` harness; runs are timestamped under `data/runs/`.

### Retrieval layer

Run [`20260829T133808Z`](data/runs/eval/20260829T133808Z.json) — all 1 000 CUAD questions, `chunks` table, `BAAI/bge-m3` (1024-d), recursive chunks (1000 chars / 100 overlap), **hybrid (vector + BM25 → RRF)**, reranked by `bge-reranker-v2-m3` over a 60-candidate pool:

| k | 1 | 3 | 5 | 10 | 20 |
|---|:---:|:---:|:---:|:---:|:---:|
| **hit@k** | **0.541** | 0.698 | 0.819 | 0.865 | **0.904** |
| **mrr@k** | **0.541** | 0.614 | 0.642 | 0.648 | 0.650 |

`recall@5 = 0.752` · `precision@5 = 0.233` · 1 639 ms/query.

Two things this table is actually good for:

- **MRR flattens after @5** (0.642 → 0.650) while **hit still climbs 0.039 from @10 to @20**. The two curves having different shapes says the golds rescued late all rank *low* — they reach the context window but never the top of it. That predicted, correctly, that raising `generate_k` from 8 to 20 would convert a 5.0-point retrieval gain into only ~2.1 points end-to-end.
- **`hit@k` at the serving `generate_k` is the system's real ceiling**, not `hit@10`. When `generate_k` was hardcoded to 8 the true ceiling was `hit@8 = 0.854`, which no number in this README was reporting.

> Run-to-run noise here is tiny — four same-config runs put `hit@10` at exactly 0.865 and `hit@1` within ±0.0035 — because there is no LLM in this loop. The *judge* metrics further down have a ±0.017 band and must not be read with the same eye.

### Generation layer — iteration history

Headline metric is the **semantic-hit rate** (does the answer actually deliver the gold answer). 200 questions per run.

| Version | Generator | semantic-hit ↑ | false-answer ↓ | false-refusal ↓ | latency |
|---------|-----------|:---:|:---:|:---:|:---:|
| v1 baseline | Gemini | 0.680 | — | 0.120 | — |
| v2 few-shot, k=10 | Gemini | 0.740 | — | **0.040** | — |
| v3 verbatim-quote constraint | Gemini | 0.760 | — | 0.100 | — |
| v4 + `doc_meta` injection | Gemini | 0.820 | 0.200 | **0.040** | 756 ms |
| v5 provider migration, `thinking=on` | GLM-4.7-Flash | 0.800 | 0.120 | 0.060 | 7 774 ms |
| **current** — `thinking=off`, `top_k=20` | **Qwen3.7-Flash** | **0.880**¹ | **0.060** | 0.120 | 1 579 ms |

¹ **The ruler changed with the last row, so read that cell carefully.** v1–v5 scored a hit as *whole-answer vs. whole-gold cosine ≥ 0.70*, which systematically punished the behaviour the prompt demands: the generator is told to *quote the exact sentence(s)*, while CUAD golds are short spans cut out of those sentences — a 40-word quote containing the gold **verbatim** scores ~0.5. The criterion is now *verbatim containment **or** cosine*. Under the old cosine-only ruler the current row is **0.780**; both numbers are stored in every run file. The acceptance test for that change was not the +0.080 — it was that the *unchanged* cosine column stayed put under paired comparison (net −1 of 200, p = 1.000).

Two results worth more than the table:

- **On the 50 answerable questions, there are zero wrong answers.** Once the ruler was fixed, all 6 misses turned out to be refusals — `semantic_hit_rate = 1 − false_refusal_rate` exactly, 44 + 6 = 50. The old ruler had been scoring "declined to answer" and "answered wrong" as the same thing, so 0.800 could not tell you which one you had. The remaining hallucination surface is on the *other* subset: 9 of the 150 unanswerable questions were answered anyway.
- **Turning `thinking` off cost nothing measurable and saved 9× latency** (7 927 → 896 ms, 200/200 questions faster, output tokens −97%). Quality moved less than the gap between two same-config replicates, so the call was made on the product axis instead: it refuses slightly more and invents slightly less, which is the safe side for legal QA.

> ⚠️ RAGAS-style `faithfulness` / `answer_relevancy` are **deliberately not in this table**. The judge model changed during the migration, and those two metrics are judge-relative — v4's 0.667/0.967 and v5's 0.500/0.857 are two different rulers, not a regression. `false-answer`, `false-refusal` and `semantic-hit` don't depend on the judge, which is why they are the ones shown.

### OCR layer (independent)

MinerU on **OmniDocBench** (1615 samples). These numbers come from a self-hosted `mineru-api`
(`hybrid-auto-engine`) that has since been replaced by MinerU's hosted API — the hosted API
exposes `model_version` instead of `backend`, so the baseline needs a re-run to stay comparable:

| | CER | WER |
|--|:---:|:---:|
| **Overall** | **7.35%** | **9.22%** |
| academic_literature | 11.02% | 12.24% |
| book | 13.51% | 15.77% |
| research_report | 27.24% | 39.96% |

> `research_report`'s gap is a **multi-column reading-order** problem (WER ≫ CER), not character recognition — a useful diagnostic the metric split makes visible.

---

## Tech stack

| Layer | Choice |
|-------|--------|
| Language / tooling | Python 3.11+, [`uv`](https://github.com/astral-sh/uv), `ruff` |
| Vector store | PostgreSQL + **pgvector** (dense) + **tsvector** (BM25 full-text) |
| Embeddings | `BAAI/bge-m3` (1024-d) via an OpenAI-compatible API (SiliconFlow) |
| Reranker | BAAI/bge-reranker-v2-m3 (TEI `/v1/rerank`) |
| Generation & judging | `qwen3.7-flash` via DashScope, OpenAI-compatible (JSON mode for generation). The eval judge is `qwen3.7-plus` — a **different** model, so the system never grades its own homework |
| Serving | FastAPI (async, SSE) + Gradio, single process |
| OCR | MinerU hosted API (v4 batch: presigned upload → poll → zip), benchmarked on OmniDocBench |
| Data | CUAD (HuggingFace `theatticusproject/cuad`) |
| Infrastructure as code | **Terraform** — `import`-based adoption of the existing AWS estate, `plan` as the change gate (see [`infra/`](infra/)) |

---

## Quickstart

### Prerequisites

1. **PostgreSQL with the `pgvector` extension** (`CREATE EXTENSION vector;`). Schema and indexes are created automatically on first ingest.
2. **An embedding endpoint and a reranker endpoint** — both ship pointed at [SiliconFlow](https://siliconflow.cn) (`BAAI/bge-m3` + `BAAI/bge-reranker-v2-m3`), so all you need is `EMBED_API_KEY` in `.env`. To use a different provider, change `embedding.base_url` / `reranker.base_url` in `config.yaml`; the embedding side expects an OpenAI-compatible `/v1/embeddings`. For a self-hosted GPU behind SSH, set `provider: ssh_tunnel` and the app opens the port-forward for you.
3. **A DashScope API key** (`GENERATE_MODEL_API`) — used for generation, judging, and the optional Contextual-RAG / HyDE / agentic features. Any OpenAI-compatible chat endpoint works: change `contextual.base_url` / `contextual.model` in `config.yaml`.
4. **A MinerU API token** (`MINERU_API_TOKEN`) — only for the OCR scripts; create one at [mineru.net/apiManage](https://mineru.net/apiManage).

### Install

```bash
uv pip install -e .
```

### Configure

Copy `.env.example` → `.env` and fill in:

```env
EMBED_API_KEY=...     # embedding service auth
PG_PASSWORD=...       # PostgreSQL password
GENERATE_MODEL_API=...  # generation, judging, and optional LLM-based features
```

All runtime parameters live in **`config.yaml`**. CLI flags (`--overlap`, `--table`, `--reranker`, …) override the YAML **at runtime without editing the file**.

### Ingest & serve

```bash
# 1. Chunk, embed and store the corpus (CUAD is fetched from HuggingFace on first run)
uv run scripts/ingest.py

# 2. Launch the API + Gradio UI
uv run scripts/serve.py            # http://127.0.0.1:6800/ui
uv run scripts/serve.py --no-ui    # REST API only
```

### Query the API

```bash
curl -s http://127.0.0.1:6800/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the governing law of this contract?",
       "doc_id": null,
       "top_k": 20,
       "generate_k": 0,
       "agentic": false}'
```

```jsonc
{
  "question": "What is the governing law of this contract?",
  "answer": "This Agreement is governed by the laws of the State of California...",
  "refused": false,
  "citations": [
    {"doc_id": "...", "chunk_id": "...", "start": 1843, "end": 1971, "excerpt": "..."}
  ],
  "latency_ms": 1579.2
}
```

`top_k` and `generate_k` both default to the value in `config.yaml` (currently 20); `generate_k: 0` means *follow `top_k`*. Set `"stream": true` for token-by-token SSE, or `"agentic": true` for the sufficiency-judge loop, which re-retrieves with a different strategy when the first pass is judged insufficient.

### CLI

```bash
legal-rag ingest --docs-dir data/cuad_docs
legal-rag query  --question "What is the term of this agreement?" --k 5
legal-rag eval   --qa-file data/qa_cuad.jsonl --output data/runs/eval.json
```

### Reproduce the evaluations

```bash
uv run scripts/eval.py --reranker                              # retrieval metrics
uv run scripts/eval_generation.py --limit 200 --reranker \
    --sim-threshold 0.70 --ragas --ragas-limit 30              # generation metrics + RAGAS
uv run scripts/grid_search.py --reranker                       # hyperparameter sweep
```

---

## Configuration highlights (`config.yaml`)

Every retrieval/generation strategy is a toggle, which is what makes the ablation studies possible:

```yaml
retrieval:   { mode: hybrid, top_k: 20, rerank_top_k: 60 }   # vector | bm25 | hybrid
reranker:    { enabled: true, model: BAAI/bge-reranker-v2-m3, tpm_limit: 450000 }
contextual:  { enabled: false, model: qwen3.7-flash, thinking: false }  # generation + Contextual RAG
hyde:        { enabled: false }                               # hypothetical-document embeddings
multi_query: { enabled: false, n: 3 }                         # query expansion + RRF
parent_child:{ parent_chars: 1000, child_chars: 300 }         # small-to-big retrieval
```

---

## Project structure

```
lex-rag/
├── lex_rag/                 # core package (~4.7k LOC)
│   ├── pipeline.py          # the single orchestrator: ingest + query paths
│   ├── store.py             # pgvector + BM25 store, dynamic table names, auto-schema
│   ├── strategy.py          # RetrievalStrategy — retrieval config as a runtime argument
│   ├── chunking.py          # recursive & parent-child chunkers
│   ├── embeddings.py        # OpenAI-compatible client with on-disk cache
│   ├── reranker.py          # cross-encoder rerank client, client-side TPM rate limiting
│   ├── contextualizer.py    # Contextual RAG / HyDE / multi-query / metadata extraction
│   ├── generator.py         # JSON-mode generator with refusal gate + citations
│   ├── llm.py               # single entry point for every LLM call (OpenAI-compatible)
│   ├── agent.py             # agentic loop: pick strategy → retrieve → accumulate → judge
│   ├── sufficiency.py       # the judge: "is this enough to answer?" + what is missing
│   ├── trace_sink.py        # per-round JSONL experiment corpus (fsync'd line by line)
│   ├── tracing.py           # Langfuse wrapper — a complete no-op when unconfigured
│   ├── evals.py             # retrieval metric computation
│   └── config.py            # dataclass config, YAML + .env loader
├── scripts/                 # ingest / serve / eval / grid-search / OCR entrypoints
├── tests/                   # 241 tests, no external services required
├── docs/                    # baseline.md, experiments.md, bug_fixes.md,
│                            # refactor_regression.md, agentic_loop_upgrade.md
├── infra/                   # Terraform (IaC) — CloudWatch log group + ECS task definition
│                            # under management; RDS/EC2/ALB deliberately out of scope
├── config.yaml              # all runtime parameters
└── CLAUDE.md                # detailed engineering log & full run history
```

---

## Design decisions worth calling out

- **Refusal as a structured field, not a heuristic.** JSON mode returns `{"refused": bool, ...}`, eliminating the "soft refusal" ambiguity where a model hedges in prose. The eval harness then measures false-refusal and false-answer rates directly. (Note: an OpenAI-compatible `json_object` mode guarantees valid syntax, not a schema — the parsers field defaults are the backstop.)
- **BM25 inside PostgreSQL.** Full-text search uses a `GENERATED` `tsvector` column, so dense and sparse retrieval hit the same store with one connection — no separate search engine to operate. (A subtle CUAD-template bug that broke BM25 OR-semantics is documented in `docs/bug_fixes.md`.)
- **Config-driven ablations.** Scripts override config via `dataclasses.replace()` at runtime rather than mutating files, so a grid search can fan out across parameter combinations without side effects.
- **Provenance tracking.** An `ingest_meta` table records the actual chunking parameters used for each table, so the evaluator reads real ingest settings instead of trusting the current YAML.
- **Lazy, optional heavy deps.** The OpenAI client is built on first use inside `ChatClient` — importing a module never opens a connection or reads a key.
- **`autocommit=True` on the store connection, on purpose.** psycopg opens an implicit transaction on the first statement, and a read-only query never commits — so a long-lived `serve.py` would sit permanently `idle in transaction`, holding a table lock (another process's `ALTER TABLE` blocks *forever, silently*) and pinning a snapshot so `VACUUM` cannot reclaim dead tuples. Neither symptom is visible to a functional test: the queries keep returning correct results. Pinned by `tests/test_store_transactions.py`.
- **Tests that pin *configuration*, not just code.** Three separate incidents had the same shape: a value moved in `config.yaml` and its reader did not follow — `serve.py` hardcoding `top_k=10`, `reranker.enabled` sitting at `false` so production ran a configuration that had never been benchmarked, and the eval report hardcoding `hit@1/3/5/10`. All three were completely silent; the system kept answering. `tests/test_serve_defaults.py` and `tests/test_eval_report.py` now assert that the served path *is* the benchmarked path.

---

## Roadmap

Honest next steps to take this from "strong portfolio project" to "deployable service":

- [x] Unit tests (`pytest`) for the pure logic — 241 of them, none requiring Postgres, an embedding endpoint or an LLM.
- [x] `docker-compose` (Postgres + pgvector) for one-command reproducibility.
- [x] CI (GitHub Actions: `ruff` + `pytest`).
- [x] Wire the OCR pipeline's output directly into the RAG ingest path end-to-end (`scripts/ingest_ocr.py`).
- [ ] **API auth / rate-limiting / structured request logging** — `POST /query` is currently unauthenticated and has no `request_id` in its logs. The next piece of work.
- [ ] **An end-to-end OCR→RAG demo doc** — the code path exists; the two-minute narrated version of it does not.
- [ ] **Release gates** — a 10–20 case regression set (answerable / unanswerable / metadata-dependent / prompt-injection) with a false-answer threshold that blocks a merge.

---

## Acknowledgements

- **CUAD** — The Atticus Project, contract-understanding dataset.
- **OmniDocBench** — OpenDataLab, OCR benchmark.
- **MinerU** — document-parsing engine used for the OCR layer.
