"""
OCR 逐样本对比工具。

每种文档类型取第 1 个样本，运行 OCR 后将 GT 与识别结果并排写入
data/runs/ocr_review/<ts>.md，供人工 review。

用法：
    uv run scripts/review_ocr.py --api-url http://127.0.0.1:6006
    uv run scripts/review_ocr.py --api-url http://127.0.0.1:6006 --backend hybrid-auto-engine
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import PIL.Image
import editdistance
import httpx
from tqdm import tqdm

PIL.Image.MAX_IMAGE_PIXELS = None

OUT_DIR = Path("data/runs/ocr_review")

TEXT_CATS = {"text_block", "header", "figure_caption", "table_caption",
             "page_footer", "page_header"}


# ---------------------------------------------------------------------------
# 复用 eval_ocr 的核心函数
# ---------------------------------------------------------------------------

def pil_to_png_bytes(img: PIL.Image.Image) -> bytes:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _extract_markdown(resp_json: dict) -> str:
    results = resp_json.get("results", {})
    if isinstance(results, dict):
        for file_result in results.values():
            if isinstance(file_result, dict):
                v = file_result.get("md_content", "")
                if v:
                    return "\n".join(v) if isinstance(v, list) else str(v)
    return ""


def ocr_one(png_bytes: bytes, filename: str, api_url: str, client: httpx.Client,
            lang: str = "ch", backend: str = "hybrid-auto-engine") -> str:
    files = [
        ("files",               (filename, png_bytes, "image/png")),
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


def cer(pred: str, gt: str) -> float:
    if not gt:
        return 0.0
    return editdistance.eval(list(pred), list(gt)) / len(gt)


# ---------------------------------------------------------------------------
# 加载每类第 1 个样本
# ---------------------------------------------------------------------------

def load_one_per_type(api_url: str) -> list[dict]:
    from datasets import load_dataset

    anno_url   = "https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/main/OmniDocBench.json"
    anno_cache = Path("data/omnidocbench_annotations.json")
    anno_cache.parent.mkdir(parents=True, exist_ok=True)
    if not anno_cache.exists():
        print("下载标注文件 ...")
        urllib.request.urlretrieve(anno_url, anno_cache)

    with open(anno_cache, encoding="utf-8") as f:
        annotations: list[dict] = json.load(f)

    print("加载 OmniDocBench 图像 ...")
    ds = load_dataset("opendatalab/OmniDocBench", split="train")

    img_name_to_idx: dict[str, int] = {}
    for i in range(len(ds)):
        img = ds[i]["image"]
        fname = Path(img.filename).name if hasattr(img, "filename") and img.filename else None
        if fname:
            img_name_to_idx[fname] = i

    seen_types: set[str] = set()
    samples = []

    for idx, anno in enumerate(annotations):
        page_info = anno.get("page_info", {})
        attr      = page_info.get("page_attribute", {})
        dtype     = attr.get("data_source", "unknown").lower().replace(" ", "_")

        if dtype in seen_types:
            continue

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

        pil_img = ds[hf_idx]["image"]
        pil_img.load()

        seen_types.add(dtype)
        samples.append({"image": pil_img, "gt_text": gt_text, "doc_type": dtype,
                         "image_name": img_name})

    print(f"共 {len(samples)} 种类型，每类 1 个样本")
    return samples


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:6006")
    parser.add_argument("--backend", default="hybrid-auto-engine",
                        choices=["pipeline", "hybrid-auto-engine", "vlm"])
    parser.add_argument("--lang", default="ch")
    args = parser.parse_args()

    samples = load_one_per_type(args.api_url)

    lines: list[str] = [
        f"# OCR Review — {args.backend}",
        f"生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    with httpx.Client(timeout=httpx.Timeout(300.0), follow_redirects=True) as client:
        health = client.get(f"{args.api_url.rstrip('/')}/health")
        health.raise_for_status()

        for item in tqdm(samples, desc="Review OCR"):
            dtype = item["doc_type"]
            png_bytes = pil_to_png_bytes(item["image"])

            try:
                pred_md = ocr_one(png_bytes, f"{dtype}_review.png",
                                  args.api_url, client,
                                  lang=args.lang, backend=args.backend)
            except Exception as e:
                pred_md = f"[OCR 失败: {e}]"

            gt = item["gt_text"]
            score = cer(pred_md, gt)

            lines += [
                f"---",
                f"## {dtype}",
                f"图像：`{item['image_name']}`　CER: **{score:.4f}**",
                "",
                "### Ground Truth",
                "```",
                gt[:2000] + ("..." if len(gt) > 2000 else ""),
                "```",
                "",
                "### OCR 输出",
                "```",
                pred_md[:2000] + ("..." if len(pred_md) > 2000 else ""),
                "```",
                "",
            ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"{ts}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
