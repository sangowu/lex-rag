import argparse
from pathlib import Path

from legal_rag_v1.config import load_config
from legal_rag_v1.pipeline import RAGPipeline
from legal_rag_v1.cuad import load_qa
from legal_rag_v1.evals import evaluate, save_result


def cmd_ingest(args, pipeline: RAGPipeline) -> None:
    docs_dir = Path(args.docs_dir)
    print(f"Ingesting documents from {docs_dir} ...")
    pipeline.ingest(docs_dir)
    print("Done.")


def cmd_query(args, pipeline: RAGPipeline) -> None:
    chunks = pipeline.query(
        args.question,
        doc_id=args.doc_id or None,
        k=args.k,
    )
    print(f"Retrieved {len(chunks)} chunk(s):\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}] doc={chunk.doc_id}  pos={chunk.start}-{chunk.end}")
        print(f"    {chunk.text[:200]}")
        print()


def cmd_eval(args, pipeline: RAGPipeline, cfg) -> None:
    qa_items = load_qa(Path(args.qa_file))
    print(f"Evaluating {len(qa_items)} QA items ...")
    result = evaluate(pipeline, qa_items, cfg.evaluation)

    print("\nhit@k :")
    for k, v in result.hit_at_k.items():
        print(f"  k={k:>2}  {v:.4f}")
    print("mrr@k :")
    for k, v in result.mrr_at_k.items():
        print(f"  k={k:>2}  {v:.4f}")

    out = Path(args.output)
    save_result(result, out)
    print(f"\nSaved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="legal-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Chunk, embed and store documents")
    p_ingest.add_argument("--docs-dir", default="data/cuad_docs")

    p_query = sub.add_parser("query", help="Retrieve relevant chunks for a question")
    p_query.add_argument("--question", required=True)
    p_query.add_argument("--doc-id", default=None)
    p_query.add_argument("--k", type=int, default=None)

    p_eval = sub.add_parser("eval", help="Run retrieval evaluation over a QA file")
    p_eval.add_argument("--qa-file", default="data/qa_cuad.jsonl")
    p_eval.add_argument("--output", default="data/runs/eval.json")

    parser.add_argument("--refresh-cache", action="store_true", help="Discard embedding cache and re-embed from scratch")
    args = parser.parse_args()
    cfg = load_config()
    pipeline = RAGPipeline(cfg, refresh_cache=args.refresh_cache)

    try:
        if args.cmd == "ingest":
            cmd_ingest(args, pipeline)
        elif args.cmd == "query":
            cmd_query(args, pipeline, )
        elif args.cmd == "eval":
            cmd_eval(args, pipeline, cfg)
    finally:
        pipeline.close()
