"""
Legal RAG 服务：FastAPI（REST API）+ Gradio UI，共享同一 pipeline 实例。

启动：
    uv run scripts/serve.py                    # API + UI（默认）
    uv run scripts/serve.py --host 0.0.0.0 --port 6800
    uv run scripts/serve.py --no-ui            # 仅 API

端点：
    POST /query          — 单文档或 corpus 问答（支持流式 SSE、Agentic）
    GET  /health         — 健康检查
    GET  /ui             — Gradio 问答界面（--no-ui 时不可用）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from legal_rag_v1._shared import get_generator, get_pipeline
from legal_rag_v1.generator import GenerationResult

# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

app = FastAPI(title="Legal RAG API", version="1.0.0")

# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    doc_id: str | None = Field(None, description="指定合同 ID；为 None 时进行 corpus 全库检索")
    top_k: int = Field(10, ge=1, le=50)
    generate_k: int = Field(8, ge=1, le=50)
    stream: bool = Field(False, description="是否流式返回答案（SSE）")
    agentic: bool = Field(False, description="是否启用 Agentic 迭代检索")


class CitationOut(BaseModel):
    doc_id: str
    chunk_id: str
    start: int | None
    end: int | None
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    citations: list[CitationOut]
    latency_ms: float
    query_trace: list[str] = []
    error: str | None = None


# ---------------------------------------------------------------------------
# 核心查询逻辑（同步）
# ---------------------------------------------------------------------------

def _run_query(req: QueryRequest) -> QueryResponse:
    pipeline = get_pipeline()
    generator = get_generator()

    if req.agentic:
        from legal_rag_v1.agent import AgenticPipeline
        from legal_rag_v1.config import load_config
        agent = AgenticPipeline(pipeline, load_config().contextual)
        chunks, query_trace = agent.query(req.question, doc_id=req.doc_id, k=req.top_k)
    else:
        chunks = pipeline.query(req.question, k=req.top_k, doc_id=req.doc_id)
        query_trace = [req.question]

    if not chunks:
        return QueryResponse(
            question=req.question, answer="", refused=True,
            citations=[], latency_ms=0.0, query_trace=query_trace,
        )

    metas = pipeline.get_doc_metas_for_chunks(chunks)
    gen_chunks = chunks[:req.generate_k]
    is_corpus = req.doc_id is None

    result = generator.generate(
        req.question, gen_chunks,
        metas=metas if is_corpus else None,
        meta=metas.get(req.doc_id) if (not is_corpus and metas) else None,
    )

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        refused=result.is_refused,
        citations=[CitationOut(
            doc_id=c.doc_id, chunk_id=c.chunk_id,
            start=c.start, end=c.end, excerpt=c.excerpt,
        ) for c in result.citations],
        latency_ms=result.latency_ms,
        query_trace=query_trace,
        error=result.error,
    )


# ---------------------------------------------------------------------------
# 流式生成（SSE）
# ---------------------------------------------------------------------------

async def _stream_query(req: QueryRequest):
    loop = asyncio.get_event_loop()
    pipeline = get_pipeline()
    generator = get_generator()

    if req.agentic:
        from legal_rag_v1.agent import AgenticPipeline
        from legal_rag_v1.config import load_config
        agent = AgenticPipeline(pipeline, load_config().contextual)
        chunks, query_trace = await loop.run_in_executor(
            None, lambda: agent.query(req.question, doc_id=req.doc_id, k=req.top_k)
        )
    else:
        chunks = await loop.run_in_executor(
            None, lambda: pipeline.query(req.question, k=req.top_k, doc_id=req.doc_id)
        )
        query_trace = [req.question]

    yield f"data: {json.dumps({'query_trace': query_trace})}\n\n"

    if not chunks:
        yield f"data: {json.dumps({'refused': True, 'citations': []})}\n\n"
        yield "data: [DONE]\n\n"
        return

    metas = pipeline.get_doc_metas_for_chunks(chunks)
    gen_chunks = chunks[:req.generate_k]
    is_corpus = req.doc_id is None

    final_result = None
    for item in generator.generate_stream(
        req.question, gen_chunks,
        metas=metas if is_corpus else None,
        meta=metas.get(req.doc_id) if (not is_corpus and metas) else None,
    ):
        if isinstance(item, GenerationResult):
            final_result = item
        else:
            yield f"data: {json.dumps({'token': item})}\n\n"
        await asyncio.sleep(0)

    if final_result:
        citations = [{"doc_id": c.doc_id, "chunk_id": c.chunk_id,
                      "start": c.start, "end": c.end, "excerpt": c.excerpt}
                     for c in final_result.citations]
        yield f"data: {json.dumps({'refused': final_result.is_refused, 'citations': citations, 'latency_ms': final_result.latency_ms})}\n\n"

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui")


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if req.stream:
        return StreamingResponse(_stream_query(req), media_type="text/event-stream")
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run_query, req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_CORPUS_LABEL = "— Corpus（全库）—"


def _get_doc_ids() -> list[str]:
    for d in ["data/cuad_docs", "data/contracts", "data/legalbench_docs"]:
        p = Path(d)
        if p.exists():
            ids = sorted(f.stem for f in p.glob("*.txt"))
            if ids:
                return ids
    return []


def _ui_query(
    message: str,
    history: list,  # noqa: ARG001 — required by Gradio ChatInterface signature
    doc_id: str,
    top_k: int,
    generate_k: int,
    use_agentic: bool,
):
    if not message or not message.strip():
        yield "请输入问题。"
        return

    pipeline = get_pipeline()
    generator = get_generator()
    query_doc_id = doc_id if doc_id and doc_id != _CORPUS_LABEL else None

    retrieval_log: list[str] = []

    if use_agentic:
        from legal_rag_v1.agent import AgenticPipeline
        from legal_rag_v1.config import load_config
        agent = AgenticPipeline(pipeline, load_config().contextual)
        chunks, query_trace = [], [message]
        for event in agent.query_stream(message, doc_id=query_doc_id, k=top_k):
            if isinstance(event, str):
                retrieval_log.append(event)
                yield "\n".join(retrieval_log)
            else:
                chunks, query_trace = event
    else:
        query_trace = [message]
        retrieval_log.append("⏳ 检索中...")
        yield "\n".join(retrieval_log)
        chunks = pipeline.query(message, k=top_k, doc_id=query_doc_id)
        msg = f"✅ 找到 {len(chunks)} 个相关片段，开始生成..." if chunks else "⚠️ 未找到相关内容"
        retrieval_log.append(msg)
        yield "\n".join(retrieval_log)

    if not chunks:
        yield "⚠️ 未找到相关合同内容，无法回答。"
        return

    metas = pipeline.get_doc_metas_for_chunks(chunks)
    gen_chunks = chunks[:generate_k]

    accumulated = ""
    final_result: GenerationResult | None = None

    for item in generator.generate_stream(
        message, gen_chunks,
        metas=metas if query_doc_id is None else None,
        meta=metas.get(query_doc_id) if (query_doc_id and metas) else None,
    ):
        if isinstance(item, GenerationResult):
            final_result = item
        else:
            accumulated += item
            yield accumulated

    # 检索日志作为折叠前缀追加到最终输出
    if use_agentic and len(query_trace) > 1:
        retrieval_summary = f"\n\n*迭代查询：{' → '.join(query_trace)}*"
    else:
        retrieval_summary = ""

    if final_result is None:
        yield "❌ 生成失败"
        return
    if final_result.error:
        yield f"❌ 错误：{final_result.error}"
        return
    if final_result.is_refused:
        yield f"🚫 合同中未找到相关信息，拒绝回答。{retrieval_summary}"
        return

    output = final_result.answer
    if final_result.citations:
        output += "\n\n---\n**引用来源：**\n"
        for c in final_result.citations:
            num = c.num if c.num else "?"
            output += (
                f"\n**[{num}]** `{c.doc_id}`  "
                f"位置 {c.start}–{c.end}\n"
                f"> {c.excerpt}...\n"
            )
    output += f"\n\n*延迟：{final_result.latency_ms:.0f} ms*{retrieval_summary}"
    yield output


def _ui_upload(file, doc_dropdown):
    import gradio as gr
    if file is None:
        return "未选择文件", doc_dropdown
    src = Path(file.name)
    dest_dir = Path("data/cuad_docs")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy(src, dest)
    try:
        get_pipeline().ingest_document(dest)
        status = f"✅ {dest.stem} 上传并 ingest 完成"
    except Exception as e:
        status = f"❌ ingest 失败：{e}"
    new_ids = [_CORPUS_LABEL] + _get_doc_ids()
    return status, gr.Dropdown(choices=new_ids, value=dest.stem)


def build_ui():
    import gradio as gr

    doc_ids = [_CORPUS_LABEL] + _get_doc_ids()

    with gr.Blocks(title="Legal RAG") as demo:
        gr.Markdown("# ⚖️ Legal Contract Q&A")
        gr.Markdown("基于合同原文的问答系统，答案均附逐字引用来源。")

        with gr.Row():
            with gr.Column(scale=1):
                doc_dropdown = gr.Dropdown(
                    choices=doc_ids, value=_CORPUS_LABEL,
                    label="合同范围", info="选择单份合同或在全库中检索",
                )
                with gr.Accordion("高级参数", open=False):
                    top_k_slider = gr.Slider(1, 20, value=10, step=1, label="检索 top_k")
                    generate_k_slider = gr.Slider(1, 20, value=8, step=1, label="生成 generate_k")
                    agentic_toggle = gr.Checkbox(
                        value=False, label="Agentic 迭代检索",
                        info="无结果时自动重写查询并重试（最多 2 次）",
                    )
                with gr.Accordion("上传新合同", open=False):
                    upload_btn = gr.UploadButton(
                        "📎 选择 .txt 文件", file_types=[".txt"], file_count="single",
                    )
                    upload_status = gr.Textbox(label="上传状态", interactive=False, max_lines=2)

            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=_ui_query,
                    additional_inputs=[doc_dropdown, top_k_slider, generate_k_slider, agentic_toggle],
                )

        upload_btn.upload(
            _ui_upload,
            inputs=[upload_btn, doc_dropdown],
            outputs=[upload_status, doc_dropdown],
        )

        # 空输入时禁用提交按钮
        demo.load(
            fn=None,
            js="""
            () => {
                function wire() {
                    document.querySelectorAll('textarea').forEach(ta => {
                        if (ta._submitGuard) return;
                        ta._submitGuard = true;
                        const update = () => {
                            let el = ta.parentElement;
                            for (let i = 0; i < 6; i++) {
                                if (!el) break;
                                el.querySelectorAll('button').forEach(btn => {
                                    if (btn.classList.contains('primary') || btn.type === 'submit') {
                                        btn.disabled = !ta.value.trim();
                                    }
                                });
                                el = el.parentElement;
                            }
                        };
                        ta.addEventListener('input', update);
                        update();
                    });
                }
                new MutationObserver(wire).observe(document.body, {childList: true, subtree: true});
                setTimeout(wire, 800);
            }
            """,
        )

    return demo


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6800)
    parser.add_argument("--no-ui", action="store_true", help="不挂载 Gradio UI，仅提供 API")
    args = parser.parse_args()

    if not args.no_ui:
        import gradio as gr
        demo = build_ui()
        app_with_ui = gr.mount_gradio_app(app, demo, path="/ui")
        uvicorn.run(app_with_ui, host=args.host, port=args.port, timeout_graceful_shutdown=3)
    else:
        uvicorn.run(app, host=args.host, port=args.port, timeout_graceful_shutdown=3)
