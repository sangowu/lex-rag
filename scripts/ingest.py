"""
一键准备数据：下载 CUAD → 写文档文件 → 嵌入入库

用法:
    uv run scripts/ingest.py [--limit N] [--docs-dir PATH] [--qa-file PATH]

默认 limit=1000（约 25 份合同）。首次运行会从 HuggingFace 下载 CUAD_v1.json。
"""
import argparse
from dataclasses import replace
from pathlib import Path

from lex_rag.config import load_config
from lex_rag.cuad import build_qa_from_hf
from lex_rag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CUAD documents into vector store")
    parser.add_argument("--limit",          type=int, default=1000,       help="Max QA items to load (controls doc count)")
    parser.add_argument("--docs-dir",       default="data/cuad_docs",     help="Directory to write contract .txt files")
    parser.add_argument("--qa-file",        default="data/qa_cuad.jsonl", help="Output path for QA JSONL")
    parser.add_argument("--no-truncate",    action="store_true",          help="Skip truncating the chunks table")
    parser.add_argument("--contextual",     action="store_true",          help="为每个 chunk 调用 Gemini 生成上下文前缀（Contextual RAG）")
    parser.add_argument("--contextual-mode",
                        choices=["standard", "hierarchical"],
                        default=None,
                        help="standard=逐 chunk 调用 Gemini（默认），hierarchical=先切 section 再复用摘要（~10x 减少 API 调用）")
    parser.add_argument("--chunk-mode",
                        choices=["standard", "parent_child"],
                        default=None,
                        help="standard=普通 chunk（默认），parent_child=大小双层 chunk 结构")
    parser.add_argument("--meta-extract",   action="store_true",          help="ingest 时为每文档调用 Gemini 提取结构化 metadata 存入 doc_meta 表")
    parser.add_argument("--table",          default=None,                 help="目标表名（不指定时自动推断）")
    parser.add_argument("--overlap",        type=int, default=None,       help="覆盖 config.yaml 中的 chunking.overlap")
    parser.add_argument("--chunk-chars",    type=int, default=None,       help="覆盖 config.yaml 中的 chunking.chunk_chars")
    parser.add_argument("--refresh-cache",  action="store_true",          help="忽略 embed cache，强制重新计算所有向量（换模型时必须使用）")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    qa_path  = Path(args.qa_file)

    cfg = load_config()

    if args.contextual:
        cfg = replace(cfg, contextual=replace(cfg.contextual, enabled=True))

    if args.contextual_mode:
        cfg = replace(cfg, contextual_mode=args.contextual_mode)

    if args.chunk_mode:
        cfg = replace(cfg, chunk_mode=args.chunk_mode)

    if args.meta_extract:
        cfg = replace(cfg, extract_meta=True)

    if args.overlap is not None or args.chunk_chars is not None:
        cfg = replace(cfg, chunking=replace(
            cfg.chunking,
            overlap=args.overlap if args.overlap is not None else cfg.chunking.overlap,
            chunk_chars=args.chunk_chars if args.chunk_chars is not None else cfg.chunking.chunk_chars,
        ))

    # 表名：优先用 --table 参数，其次按各模式自动选择
    if args.table:
        table = args.table
    elif cfg.chunk_mode == "parent_child":
        table = "chunks_parent_child"
    elif cfg.contextual_mode == "hierarchical":
        table = "chunks_hierarchical"
    elif cfg.contextual.enabled:
        table = "chunks_contextual"
    else:
        table = cfg.database.table
    cfg = replace(cfg, database=replace(cfg.database, table=table))

    print(f"Table          : {cfg.database.table}")
    print(f"Chunk mode     : {cfg.chunk_mode}")
    print(f"Chunk          : {cfg.chunking.chunk_chars} chars, overlap={cfg.chunking.overlap}, strategy={cfg.chunking.strategy}")
    if cfg.chunk_mode == "parent_child":
        pc = cfg.parent_child
        print(f"Parent-Child   : parent={pc.parent_chars}, child={pc.child_chars}, overlap={pc.overlap}")
    print(f"Contextual     : {cfg.contextual.enabled}"
          + (f"  (mode={cfg.contextual_mode}, model={cfg.contextual.model}, rpm={cfg.contextual.rpm_limit})"
             if cfg.contextual.enabled else ""))
    print(f"Meta extract   : {cfg.extract_meta}")

    # 1. 下载 CUAD，写文档文件，生成 QA JSONL
    print(f"Building QA from HuggingFace (limit={args.limit}) ...")
    items = build_qa_from_hf(docs_dir, qa_path, limit=args.limit)
    print(f"  {len(items)} QA items, documents in {docs_dir}")

    # 2. 清空旧数据 + Ingest
    pipeline = RAGPipeline(cfg, refresh_cache=args.refresh_cache)
    try:
        if not args.no_truncate:
            print(f"Truncating {cfg.database.table} ...")
            pipeline.store.truncate()
        pipeline.ingest(docs_dir)
        pipeline.store.save_meta(
            chunk_chars=cfg.chunking.chunk_chars,
            overlap=cfg.chunking.overlap,
            strategy=cfg.chunking.strategy,
            contextual=cfg.contextual.enabled,
            chunk_mode=cfg.chunk_mode,
        )
    finally:
        pipeline.close()

    print("Ingest complete.")


if __name__ == "__main__":
    main()
