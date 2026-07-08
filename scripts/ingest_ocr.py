"""
OCR → RAG 端到端 ingest 脚本。

流程：
  1. 遍历输入目录中的 PDF / 图像文件
  2. 每个文件 POST 到 MinerU /file_parse，获取 Markdown
  3. 将 Markdown 文本直接送入 RAGPipeline._ingest_one()
     （chunking → embedding → pgvector，与 ingest.py 完全相同的路径）

用法：
    uv run scripts/ingest_ocr.py --input-dir data/scanned_docs
    uv run scripts/ingest_ocr.py --input-dir data/scanned_docs --table chunks_ocr
    uv run scripts/ingest_ocr.py --input-dir data/scanned_docs --table chunks_ocr --no-truncate

支持格式：PDF、PNG、JPG、JPEG、TIFF、BMP

依赖（本地）：httpx tqdm
远端：MinerU API（通过 SSH 隧道或直连）
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import httpx
from tqdm import tqdm

SUPPORTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

MIME = {
    ".pdf":  "application/pdf",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".tif":  "image/tiff",
    ".bmp":  "image/bmp",
}


# ---------------------------------------------------------------------------
# MinerU 调用（复用 eval_ocr 的逻辑）
# ---------------------------------------------------------------------------

def _extract_markdown(resp_json: dict) -> str:
    results = resp_json.get("results", {})
    if isinstance(results, dict):
        for file_result in results.values():
            if isinstance(file_result, dict):
                v = file_result.get("md_content", "")
                if v:
                    return "\n".join(v) if isinstance(v, list) else str(v)
    return ""


def ocr_file(path: Path, api_url: str, client: httpx.Client,
             lang: str = "ch", backend: str = "hybrid-auto-engine") -> str:
    """将单个文件发送给 MinerU，返回 Markdown 文本。"""
    ext = path.suffix.lower()
    mime = MIME.get(ext, "application/octet-stream")
    file_bytes = path.read_bytes()

    if backend == "vlm":
        files = [
            ("files",    (path.name, file_bytes, mime)),
            ("backend",  (None, "vlm")),
            ("return_md", (None, "true")),
        ]
    else:
        files = [
            ("files",               (path.name, file_bytes, mime)),
            ("backend",             (None, backend)),
            ("lang_list",           (None, lang)),
            ("parse_method",        (None, "ocr")),
            ("formula_enable",      (None, "false")),
            ("table_enable",        (None, "true")),
            ("image_analysis",      (None, "false")),
            ("return_md",           (None, "true")),
            ("return_middle_json",  (None, "false")),
            ("return_model_output", (None, "false")),
            ("return_content_list", (None, "false")),
            ("return_images",       (None, "false")),
            ("response_format_zip", (None, "false")),
        ]

    url = f"{api_url.rstrip('/')}/file_parse"
    for attempt in range(5):
        resp = client.post(url, files=files)
        if resp.status_code == 409:
            time.sleep(3 * (attempt + 1))
            continue
        resp.raise_for_status()
        rj = resp.json()
        md = _extract_markdown(rj)
        if md:
            return md
        result_url = rj.get("result_url")
        if result_url:
            time.sleep(5)
            md = _extract_markdown(client.get(result_url).json())
            if md:
                return md
        return md
    resp.raise_for_status()
    return ""


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    from legal_rag_v1.config import load_config
    from legal_rag_v1.pipeline import RAGPipeline

    parser = argparse.ArgumentParser(
        description="扫描件 OCR → RAG ingest 端到端脚本"
    )
    parser.add_argument("--input-dir",  required=True,
                        help="包含 PDF/图像文件的目录")
    parser.add_argument("--api-url",    default="http://127.0.0.1:6006",
                        help="MinerU API 地址")
    parser.add_argument("--backend",    default="hybrid-auto-engine",
                        choices=["pipeline", "hybrid-auto-engine", "vlm"],
                        help="MinerU 解析后端（默认 hybrid-auto-engine）")
    parser.add_argument("--lang",       default="ch",
                        help="OCR 语言（默认 ch）")
    parser.add_argument("--table",      default=None,
                        help="目标 pgvector 表名（默认由 config.yaml 决定）")
    parser.add_argument("--no-truncate", action="store_true",
                        help="不清空现有数据，增量追加")
    parser.add_argument("--contextual",  action="store_true",
                        help="为每个 chunk 调用 Gemini 生成上下文前缀")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"输入目录不存在：{input_dir}")

    files = sorted(p for p in input_dir.iterdir()
                   if p.suffix.lower() in SUPPORTED_EXTS)
    if not files:
        raise SystemExit(f"目录中未找到支持的文件（{', '.join(SUPPORTED_EXTS)}）")

    print(f"找到 {len(files)} 个文件，目录：{input_dir}")

    # 初始化 RAG pipeline
    cfg = load_config()
    if args.contextual:
        cfg = replace(cfg, contextual=replace(cfg.contextual, enabled=True))
    if args.table:
        cfg = replace(cfg, database=replace(cfg.database, table=args.table))

    print(f"目标表：{cfg.database.table}  contextual={cfg.contextual.enabled}")

    pipeline = RAGPipeline(cfg)
    try:
        if not args.no_truncate:
            print(f"清空表 {cfg.database.table} ...")
            pipeline.store.truncate()

        with httpx.Client(timeout=httpx.Timeout(300.0), follow_redirects=True) as client:
            health = client.get(f"{args.api_url.rstrip('/')}/health")
            health.raise_for_status()
            print(f"MinerU API: {args.api_url}  version={health.json().get('version', '?')}")

            for path in tqdm(files, desc="OCR → Ingest"):
                try:
                    md = ocr_file(path, args.api_url, client,
                                  lang=args.lang, backend=args.backend)
                    if not md.strip():
                        tqdm.write(f"  ⚠️ OCR 返回空内容，跳过：{path.name}")
                        continue
                    pipeline._ingest_one(path.stem, md)
                except Exception as e:
                    tqdm.write(f"  ⚠️ 失败（{path.name}）：{e}")

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
