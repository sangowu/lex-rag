"""
进程级单例：pipeline 和 generator，供 serve.py 和 ui.py 共用同一实例。
保证同一进程中只初始化一次 VectorStore（避免 DDL 锁竞争）。
"""
from __future__ import annotations

from legal_rag_v1.config import load_config
from legal_rag_v1.generator import LegalGenerator
from legal_rag_v1.pipeline import RAGPipeline

_pipeline: RAGPipeline | None = None
_generator: LegalGenerator | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(load_config())
    return _pipeline


def get_generator() -> LegalGenerator:
    global _generator
    if _generator is None:
        _generator = LegalGenerator(load_config().contextual)
    return _generator
