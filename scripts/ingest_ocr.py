"""
OCR → RAG 端到端 ingest 脚本。

流程：
  1. 遍历输入目录中的 PDF / 图像文件
  2. 按批交给 MinerU 在线 API 解析成 Markdown
  3. 将 Markdown 文本直接送入 RAGPipeline._ingest_one()
     （chunking → embedding → pgvector，与 ingest.py 完全相同的路径）

用法：
    uv run scripts/ingest_ocr.py --input-dir data/scanned_docs
    uv run scripts/ingest_ocr.py --input-dir data/scanned_docs --table chunks_ocr
    uv run scripts/ingest_ocr.py --input-dir data/scanned_docs --table chunks_ocr --no-truncate

支持格式：PDF、PNG、JPG、JPEG、TIFF、BMP（TIFF 线上未列入支持列表，会被跳过并提示）

前置：.env 里设置 MINERU_API_TOKEN

依赖（本地）：httpx tqdm
"""
from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from lex_rag import ocr
from lex_rag.ingest_guard import Verdict, summarise

SUPPORTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp"}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    from lex_rag.config import load_config
    from lex_rag.pipeline import RAGPipeline

    parser = argparse.ArgumentParser(
        description="扫描件 OCR → RAG ingest 端到端脚本"
    )
    parser.add_argument("--input-dir",  required=True,
                        help="包含 PDF/图像文件的目录")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="每批提交给在线 API 的文件数（官方单批上限 200）")
    parser.add_argument("--table",      default=None,
                        help="目标 pgvector 表名（默认由 config.yaml 决定）")
    parser.add_argument("--no-truncate", action="store_true",
                        help="不清空现有数据，增量追加")
    parser.add_argument("--contextual",  action="store_true",
                        help="为每个 chunk 调用 Gemini 生成上下文前缀")
    parser.add_argument("--fail-on-changed-source", action="store_true",
                        help="有文档内容与上次 ingest 不同时退出码 1（默认只报告）")
    ocr.add_ocr_args(parser)
    args = parser.parse_args()
    # Windows 控制台默认 GBK，遇到 emoji 会抛 UnicodeEncodeError —— 重设为 utf-8。
    # 与 eval_experiment.py 的处理保持一致。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args.batch_size = max(1, min(args.batch_size, ocr.MAX_BATCH_FILES))
    load_dotenv()          # MINERU_API_TOKEN

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

        results = []
        opts = ocr.options_from_args(args)
        print(f"MinerU 在线 API: {ocr.MINERU_API_BASE}  model_version={opts.model_version}")

        with (
            ocr.MinerUCloudClient(options=opts) as client,
            tqdm(total=len(files), desc="OCR → Ingest") as bar,
        ):
            for start_i in range(0, len(files), args.batch_size):
                batch = files[start_i:start_i + args.batch_size]
                try:
                    mds = client.parse_paths(batch)
                except Exception as e:
                    bar.write(f"  ⚠️ 批次 {start_i}~{start_i + len(batch)} 整批失败：{e}")
                    bar.update(len(batch))
                    continue

                # ingest 逐个做：单个文档 chunk/embed 失败不该拖掉整批 OCR 结果
                for path, md in zip(batch, mds):
                    try:
                        if not md.strip():
                            bar.write(f"  ⚠️ OCR 返回空内容，跳过：{path.name}")
                            continue
                        results.append(pipeline._ingest_one(path.stem, md, source=str(path)))
                    except Exception as e:
                        bar.write(f"  ⚠️ ingest 失败（{path.name}）：{e}")
                bar.update(len(batch))

        pipeline.store.save_meta(
            chunk_chars=cfg.chunking.chunk_chars,
            overlap=cfg.chunking.overlap,
            strategy=cfg.chunking.strategy,
            contextual=cfg.contextual.enabled,
            chunk_mode=cfg.chunk_mode,
        )
    finally:
        pipeline.close()

    # OCR 这条路径比 ingest.py 更需要指纹：**同一份扫描件重跑 OCR，文本会变**
    # （`docs/demo_ocr_rag.md` 实测换个噪声实例 CER 就差 0.0017）。没有指纹的话，
    # "语料被人改写了"和"这一轮 OCR 跑歪了"在数据库里长得一模一样。
    print()
    print(summarise(results))
    if args.fail_on_changed_source and any(v is Verdict.CHANGED for _, v, _ in results):
        raise SystemExit(1)

    print("Ingest complete.")


if __name__ == "__main__":
    main()
