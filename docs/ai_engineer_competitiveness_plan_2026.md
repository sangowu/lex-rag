# Legal RAG 2026 AI Engineer Competitiveness Plan

## Summary

Goal: turn `legal_rag_v1` from a strong RAG/OCR research-style portfolio project into a production-oriented AI Engineer project that maps directly to 2026 Dublin/Ireland Applied AI Engineer, AI Software Engineer, RAG Engineer, and GenAI Platform Engineer job descriptions.

A second goal is to extend the current single-shot RAG pipeline into an agentic retrieval system, where the model can iteratively search, open, find, and summarize evidence before generating citation-grounded answers.

Current judgment: the project is already strong enough to include in the CV. To become more competitive in 2026, it needs clearer production evidence: automated tests, CI, reproducible infrastructure, API safety boundaries, structured logs, and an end-to-end OCR-to-RAG demo.

Market positioning: target Applied AI Engineer / AI Software Engineer / LLM-RAG Engineer roles rather than pure Research Scientist roles. Dublin-market job descriptions commonly ask for Python APIs, production GenAI systems, RAG, evaluation, Docker/CI/CD, cloud deployment, monitoring, and reliability.

## Workstream 1: Production Evidence Pack

What to add:

- Add unit tests for pure logic: RRF fusion, chunking, citation span validation, JSON refusal parsing, retrieval result shape, and API smoke behavior.
- Add GitHub Actions CI for `ruff`, `pytest`, and `python -m compileall`.
- Add `docker-compose.yml` for Postgres + pgvector one-command startup.
- Add structured request logging: `request_id`, latency, retrieved chunk ids, model/provider, refusal flag, citation validation result, and error type.
- Add minimal API key auth and simple rate limiting for public-facing API safety.

How to do it:

- Start with pure unit tests that do not require Postgres, Gemini, embedding service, or reranker.
- Keep integration tests optional and mark them separately so normal CI stays fast.
- Use env vars for all external services and keep secrets out of repo.
- Implement logging as structured JSON lines or standard Python logger fields before adding dashboards.

Success criteria:

- `pytest` passes locally.
- CI passes on push/PR.
- A new developer can start Postgres with one command and run a documented smoke query.
- README Roadmap items for tests, CI, docker-compose, and structured logging can be checked off.

## Workstream 2: OCR-to-RAG End-to-End Demo

What to add:

- A minimal demo path: scanned contract input -> MinerU Markdown -> RAG ingest -> question -> cited answer.
- `docs/demo_ocr_rag.md` with input description, OCR output snippet, answer JSON, citation example, and known OCR limits.
- README screenshots or GIFs for the Gradio UI, API JSON response, and OCR-to-RAG flow.
- A small refusal example showing that the system answers only when supported by contract text.

How to do it:

- Pick one small scanned sample rather than trying to demo the full benchmark.
- Store only lightweight sample outputs in docs if raw inputs are too large.
- Make the demo reproducible with one documented command or a short sequence of commands.
- Explicitly document the multi-column reading-order limitation already shown by OCR WER/CER analysis.

Success criteria:

- In an interview, the full scanned-document-to-cited-answer chain can be explained in two minutes.
- README no longer has a placeholder demo section.
- OCR is clearly integrated into the RAG story, not presented as an unrelated benchmark.

## Workstream 3: Evaluation Gates and Safety

What to add:

- A small regression eval set with 10-20 cases: answerable, unanswerable, metadata-dependent, citation-required, and prompt-injection-in-document.
- A single eval summary artifact with retrieval metrics, generation metrics, false-refusal, false-answer, faithfulness, latency, and pass/fail thresholds.
- Ablation experiments comparing single-shot RAG, multi-query RAG, agentic rewrite, open/find navigation, summarize-assisted evidence selection, and full agentic retrieval.
- Prompt-injection safety cases where malicious document text is ignored and the system still follows the legal QA task.
- README section explaining release gates for eval regressions.

How to do it:

- Reuse the existing retrieval and generation evaluation scripts.
- Add a thin summary layer rather than rewriting the evaluator.
- Treat false answer and invalid citation as release-blocking failures.
- Keep LLM-as-judge useful but not the only quality signal; combine it with deterministic citation validation and refusal checks.

Success criteria:

- The project can answer: "How do you know the system improved?"
- The project can answer: "How do you prevent hallucination or unsupported legal advice?"
- The project can quantify whether agentic tool use improves correctness or citation quality over the current single-shot baseline.
- A config/prompt change can be compared against the current baseline before being accepted.

## Workstream 4: Agentic Retrieval Layer

What to add:

- Add generic retrieval tools: `search`, `open`, `find`, and `summarize`.
- Keep `search` as a wrapper around the existing `RAGPipeline.query()`.
- Implement `find` as a narrower `search` scoped to one `doc_id`.
- Implement `open` as deterministic source-of-truth access to original document windows, returning `doc_id`, `chunk_id`, `start`, `end`, and exact text.
- Implement `summarize` only for multi-evidence or complex questions, never as the sole citation source.
- Add an agent loop that decides whether to search, open, find, summarize, answer, or refuse.

How to do it:

- Keep tool schemas generic, but add a legal contract adapter for clause vocabulary, refusal policy, and citation validation.
- Do not let the agent answer from search previews alone; final answers must use opened evidence.
- Record a tool trace for each answer.
- Add budgets: max iterations, max search calls, max opened chunks, and max evidence tokens.

Success criteria:

- The system can handle questions that require document navigation, such as termination exceptions, survival obligations, assignment restrictions, and buyer risk summaries.
- The response includes answer, citations, and tool trace.
- The agent can refuse when search/open/find do not produce explicit evidence.

## Workstream 5: CV and Portfolio Packaging

Recommended CV project title:

`Legal RAG - Contract QA with OCR, Citations, and Evaluation`

Recommended CV bullet structure:

- Built a production-style legal RAG system for contract QA with hybrid vector/BM25 retrieval, pgvector, reranking, structured JSON generation, refusal gates, and citation-grounded answers.
- Created an evaluation harness for retrieval and generation quality, tracking hit@k, MRR, semantic-hit rate, false-refusal, false-answer, faithfulness, answer relevance, latency, and regression history.
- Integrated an OCR pipeline for scanned legal documents using MinerU, benchmarked OCR quality on OmniDocBench, and connected OCR output into the RAG ingest path for cited QA.

Recommended portfolio order:

1. Legal RAG - Contract QA with OCR, Citations, and Evaluation
2. JobRadar - LLM Job Matching and Agentic Evaluation Pipeline
3. MacroLens - Financial Research Agent with Tool-Verified Reasoning
4. Privacy-Aware LLM Fine-Tuning
5. ProShot - Full-Stack AI Image Product

## Completion Target

Current completion estimate: about 70-75% as a strong portfolio project; about 50-55% as an agentic AI Engineer showcase.

After completing production evidence plus agentic retrieval ablation, expected completion rises to about 85-90% for 2026 AI Engineer competitiveness.

Remaining gap after that: real cloud deployment and long-running observability. This is useful but not required before putting the project on the CV.

## Suggested Order

1. Add tests and CI.
2. Add docker-compose reproducibility.
3. Add Agentic Retrieval tool interfaces: `search`, `open`, `find`, and `summarize`.
4. Add ablation eval: single-shot RAG vs agentic retrieval.
5. Add OCR-to-RAG demo and README screenshots.
6. Add structured logging, API key auth, and rate limiting.
7. Add regression eval gates and prompt-injection safety cases.

## Assumptions

- Target market is Dublin/Ireland first, remote-from-Ireland second.
- Target roles are applied/production AI engineering roles, not pure research roles.
- The project should remain lightweight enough to run locally.
- External embedding, reranker, and LLM services can remain configurable rather than bundled into the repo.
