"""
OmniDocBench OCR 评测脚本（MinerU 3.x）。

流程：
  1. 加载 OmniDocBench 数据集（可按 data_source 过滤）
  2. 将每个样本的 PIL 图像预加载为 PDF 字节
  3. POST 到 MinerU /file_parse，获取 Markdown 输出
  4. 与 ground truth 对比，计算 CER / WER
  5. 结果写入 data/runs/ocr_eval/<ts>.json

用法：
    uv run scripts/eval_ocr.py --api-url http://127.0.0.1:6006 --limit 50
    uv run scripts/eval_ocr.py --api-url http://127.0.0.1:6006 --limit 200
    uv run scripts/eval_ocr.py --api-url http://127.0.0.1:6006 --limit 50 --doc-types academic_literature,research_report

依赖：
    pip install editdistance datasets pillow httpx tqdm
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import PIL.Image
import editdistance
import httpx
from tqdm import tqdm

PIL.Image.MAX_IMAGE_PIXELS = None  # 关闭解压炸弹限制（数据集图像合法但超大）

OUT_DIR = Path("data/runs/ocr_eval")
# 实际 data_source 值：academic_literature / research_report / book /
#   PPT2PDF / colorful_textbook / magazine / exam_paper / newspaper / note
# 不指定时评测全部类型
DEFAULT_DOC_TYPES: set[str] = set()

TEXT_CATS = {"text_block", "header", "figure_caption", "table_caption",
             "page_footer", "page_header"}


# ---------------------------------------------------------------------------
# 数据集加载（同步，PIL 懒加载在此完成，避免 async 上下文冲突）
# ---------------------------------------------------------------------------

def load_omnidocbench(limit: int, doc_types: set[str],
                      samples_per_type: int | None = None) -> list[dict]:
    from datasets import load_dataset

    anno_url   = "https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/main/OmniDocBench.json"
    anno_cache = Path("data/omnidocbench_annotations.json")
    anno_cache.parent.mkdir(parents=True, exist_ok=True)

    if not anno_cache.exists():
        print(f"下载标注文件 → {anno_cache} ...")
        urllib.request.urlretrieve(anno_url, anno_cache)
        print("下载完成")

    with open(anno_cache, encoding="utf-8") as f:
        annotations: list[dict] = json.load(f)

    print(f"加载 OmniDocBench 图像（{len(annotations)} 条标注）...")
    ds = load_dataset("opendatalab/OmniDocBench", split="train")

    # 建立 filename → HF index 映射
    img_name_to_idx: dict[str, int] = {}
    for i in range(len(ds)):
        img = ds[i]["image"]
        fname = Path(img.filename).name if hasattr(img, "filename") and img.filename else None
        if fname:
            img_name_to_idx[fname] = i

    # per_type_count 用于 --samples-per-type 模式：记录每种类型已收集数量
    per_type_count: dict[str, int] = {}
    samples = []
    for idx, anno in enumerate(annotations):
        page_info = anno.get("page_info", {})
        attr      = page_info.get("page_attribute", {})
        dtype     = attr.get("data_source", "unknown").lower().replace(" ", "_")

        if doc_types and dtype not in doc_types:
            continue

        # --samples-per-type：该类型已够则跳过
        if samples_per_type is not None:
            if per_type_count.get(dtype, 0) >= samples_per_type:
                continue

        # ground truth：拼接文本块（排除公式/废弃块）
        gt_parts = []
        for block in anno.get("layout_dets", []):
            if block.get("ignore"):
                continue
            if block.get("category_type") not in TEXT_CATS:
                continue
            text = (block.get("text") or "").strip()
            if text:
                gt_parts.append(text)
        gt_text = " ".join(gt_parts).strip()
        if not gt_text:
            continue

        img_name = Path(page_info.get("image_path", "")).name
        hf_idx   = img_name_to_idx.get(img_name, idx if idx < len(ds) else None)
        if hf_idx is None:
            continue

        # 强制触发 PIL 懒加载，确保图像字节在进入 HTTP 循环前已完全加载
        pil_img = ds[hf_idx]["image"]
        pil_img.load()

        per_type_count[dtype] = per_type_count.get(dtype, 0) + 1
        samples.append({"image": pil_img, "gt_text": gt_text, "doc_type": dtype})
        if len(samples) >= limit:
            break

    print(f"筛选后样本数：{len(samples)}")
    return samples


# ---------------------------------------------------------------------------
# 图像转换工具
# ---------------------------------------------------------------------------

def split_columns(pil_image: PIL.Image.Image,
                  min_gap_px: int = 15,
                  white_threshold: int = 240) -> list[PIL.Image.Image]:
    """
    检测双栏布局并分割为左右两列。单栏时返回 [原图]。

    原理：垂直投影——统计每个 x 坐标有多少行是暗像素。
    在图像宽度 20%-80% 的中间区域寻找最宽的连续空白带（暗像素极少的竖列），
    以该空白带中心为分割点切图。min_gap_px 控制最小空白宽度，太窄则视为单栏。
    """
    gray = pil_image.convert("L")
    width, height = gray.size
    pixels = gray.load()

    # 每列暗像素数量
    dark = [sum(1 for y in range(height) if pixels[x, y] < white_threshold)
            for x in range(width)]

    col_threshold = height * 0.02   # 少于 2% 行有暗像素 → 空白列
    search_lo, search_hi = width // 5, width * 4 // 5

    best_start, best_width = None, 0
    cur_start, cur_width = None, 0
    for x in range(search_lo, search_hi):
        if dark[x] <= col_threshold:
            if cur_start is None:
                cur_start = x
            cur_width += 1
        else:
            if cur_width > best_width:
                best_width, best_start = cur_width, cur_start
            cur_start, cur_width = None, 0
    if cur_width > best_width:
        best_width, best_start = cur_width, cur_start

    if best_width < min_gap_px:
        return [pil_image]   # 未找到足够宽的空白带，视为单栏

    split_x = best_start + best_width // 2
    left  = pil_image.crop((0,       0, split_x, height))
    right = pil_image.crop((split_x, 0, width,   height))
    return [left, right]


def pil_to_png_bytes(pil_image: PIL.Image.Image) -> bytes:
    if pil_image.mode not in ("RGB", "L"):
        pil_image = pil_image.convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def pil_to_pdf_bytes(pil_image: PIL.Image.Image) -> bytes:
    """PIL 原生支持单页 PDF 输出，VLM 后端只识别 PDF 格式。"""
    if pil_image.mode not in ("RGB", "L"):
        pil_image = pil_image.convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="PDF", resolution=150)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 调用 MinerU API（同步 httpx.Client）
# ---------------------------------------------------------------------------

def _extract_markdown(resp_json: dict) -> str:
    # {"results": {"<filename>": {"md_content": "..."}}}
    results = resp_json.get("results", {})
    if isinstance(results, dict):
        for file_result in results.values():
            if isinstance(file_result, dict):
                v = file_result.get("md_content", "")
                if v:
                    return "\n".join(v) if isinstance(v, list) else str(v)
    return ""


def ocr_one(png_bytes: bytes, filename: str, api_url: str, client: httpx.Client,
            lang: str = "ch", backend: str = "pipeline",
            parse_method: str = "ocr") -> str:
    # 文本字段用 (None, value) 格式放进 files= 避免同时传 files+data 导致 h11 body 损坏
    if backend == "vlm":
        # VLM 后端：最简请求，不传 pipeline 专属参数
        files = [
            ("files",   (filename, png_bytes, "image/png")),
            ("backend", (None, "vlm")),
            ("return_md", (None, "true")),
        ]
    else:
        # pipeline 和 hybrid-auto-engine 共用相同参数，lang_list/parse_method 均有效
        files = [
            ("files",               (filename, png_bytes, "image/png")),
            ("backend",             (None, backend)),
            ("lang_list",           (None, lang)),
            ("parse_method",        (None, parse_method)),
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
        # 异步后端（vlm 等）：从 result_url 轮询
        result_url = rj.get("result_url")
        if result_url:
            time.sleep(5)
            poll = client.get(result_url)
            md = _extract_markdown(poll.json())
            if md:
                return md
        return md
    resp.raise_for_status()
    return ""


# ---------------------------------------------------------------------------
# 评测指标
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = re.sub(r"[#*`_~\[\]()>|\\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def cer(pred: str, gt: str) -> float:
    if not gt:
        return 0.0
    return editdistance.eval(list(pred), list(gt)) / len(gt)


def wer(pred: str, gt: str) -> float:
    gt_words = gt.split()
    if not gt_words:
        return 0.0
    return editdistance.eval(pred.split(), gt_words) / len(gt_words)


# ---------------------------------------------------------------------------
# 主评测流程
# ---------------------------------------------------------------------------

def run_eval(args) -> None:
    doc_types = set(args.doc_types.split(",")) if args.doc_types else DEFAULT_DOC_TYPES
    samples = load_omnidocbench(args.limit, doc_types,
                                samples_per_type=args.samples_per_type)
    if not samples:
        print("无可用样本，退出。")
        return

    all_results: list[dict] = []
    results_by_type: dict[str, list[dict]] = {}

    with httpx.Client(timeout=httpx.Timeout(300.0), follow_redirects=True) as client:
        health = client.get(f"{args.api_url.rstrip('/')}/health")
        health.raise_for_status()
        info = health.json()
        print(f"MinerU API: {args.api_url}  version={info.get('version', '?')}")

        for i, item in enumerate(tqdm(samples, desc="OCR 评测")):
            png_bytes = pil_to_png_bytes(item["image"])

            try:
                t0 = time.perf_counter()
                if args.backend == "vlm":
                    raw_bytes = pil_to_pdf_bytes(item["image"])
                    fname = f"sample_{i}.pdf"
                    pred_md = ocr_one(raw_bytes, fname, args.api_url, client,
                                      lang=args.lang, backend=args.backend,
                                      parse_method=args.parse_method)
                elif args.column_split:
                    # 列分割模式：检测双栏，各列分别 OCR 后拼接
                    cols = split_columns(item["image"])
                    parts = []
                    for ci, col_img in enumerate(cols):
                        col_bytes = pil_to_png_bytes(col_img)
                        col_md = ocr_one(col_bytes, f"sample_{i}_col{ci}.png",
                                         args.api_url, client,
                                         lang=args.lang, backend=args.backend,
                                         parse_method=args.parse_method)
                        parts.append(col_md)
                    pred_md = "\n\n".join(parts)
                else:
                    raw_bytes = png_bytes
                    fname = f"sample_{i}.png"
                    pred_md = ocr_one(raw_bytes, fname, args.api_url, client,
                                      lang=args.lang, backend=args.backend,
                                      parse_method=args.parse_method)
                latency_ms = (time.perf_counter() - t0) * 1000
            except Exception as e:
                tqdm.write(f"  ⚠️ OCR 失败（样本 {i}）：{e}")
                continue

            pred_norm = normalize(pred_md)
            gt_norm   = normalize(item["gt_text"])

            row = {
                "doc_type":   item["doc_type"],
                "cer":        cer(pred_norm, gt_norm),
                "wer":        wer(pred_norm, gt_norm),
                "latency_ms": round(latency_ms, 1),
            }
            all_results.append(row)
            results_by_type.setdefault(item["doc_type"], []).append(row)

    if not all_results:
        print("无有效结果。")
        return

    def summarize(rows: list[dict]) -> dict:
        n = len(rows)
        return {
            "n":              n,
            "avg_cer":        round(sum(r["cer"] for r in rows) / n, 4),
            "avg_wer":        round(sum(r["wer"] for r in rows) / n, 4),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / n, 1),
        }

    overall = summarize(all_results)
    by_type = {k: summarize(v) for k, v in results_by_type.items()}

    print("\n=== OCR Eval — OmniDocBench ===")
    print(f"  样本总数   : {overall['n']}")
    print(f"  avg CER    : {overall['avg_cer']:.4f}")
    print(f"  avg WER    : {overall['avg_wer']:.4f}")
    print(f"  avg 延迟   : {overall['avg_latency_ms']:.1f} ms")
    if by_type:
        print("\n  --- 按文档类型 ---")
        for dtype, m in sorted(by_type.items()):
            print(f"  {dtype:<30} n={m['n']:<4} CER={m['avg_cer']:.4f}  WER={m['avg_wer']:.4f}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{ts}.json"
    payload = {
        "run_id":           ts,
        "api_url":          args.api_url,
        "limit":            args.limit,
        "samples_per_type": args.samples_per_type,
        "doc_types":        list(doc_types),
        "lang":             args.lang,
        "backend":          args.backend,
        "parse_method":     args.parse_method,
        "column_split":     args.column_split,
        "overall":          overall,
        "by_type":          by_type,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url",   default="http://127.0.0.1:6006",
                        help="MinerU API 地址（经 SSH 隧道后的本地地址）")
    parser.add_argument("--limit",     type=int, default=50,
                        help="最多评测的样本数")
    parser.add_argument("--column-split", action="store_true",
                        help="启用列分割预处理：检测双栏布局并分列 OCR 后拼接，"
                             "针对 research_report 等多栏文档改善阅读顺序")
    parser.add_argument("--parse-method", default="ocr",
                        choices=["ocr", "auto", "txt"],
                        help="MinerU 解析方式（pipeline/hybrid 后端有效）。"
                             "ocr=强制 OCR（默认）；auto=自动判断单/多栏；txt=直接提取文字层")
    parser.add_argument("--samples-per-type", type=int, default=None,
                        help="每种文档类型固定取前 N 条样本（用于可复现的调优测试集）。"
                             "设置后 --limit 仍作为总量上限。")
    parser.add_argument("--doc-types", default=None,
                        help="逗号分隔的 data_source 值，默认全量。"
                             "可选: academic_literature,research_report,book,PPT2PDF,magazine,exam_paper,newspaper,note")
    parser.add_argument("--lang", default="ch",
                        help="OCR 语言（单值，默认 ch）。"
                             "可选: ch / en / ch_lite / korean / japan 等（MinerU 单语言模型）")
    parser.add_argument("--backend", default="hybrid-auto-engine",
                        choices=["pipeline", "hybrid-auto-engine", "vlm"],
                        help="MinerU 解析后端（默认 hybrid-auto-engine）。"
                             "hybrid-auto-engine 用小 VLM 辅助识别，OmniDocBench 精度 95+（需 8GB 显存）；"
                             "pipeline 精度 85+（需 4GB 显存）；vlm 为完整 VLM，需要 vllm")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
