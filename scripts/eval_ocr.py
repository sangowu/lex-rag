"""
OmniDocBench OCR 评测脚本（MinerU 在线 API v4）。

流程：
  1. 加载 OmniDocBench 数据集（可按 data_source 过滤）
  2. 按批把样本图像编码成 PNG，交给 MinerU 在线 API 解析成 Markdown
  3. 与 ground truth 对比，计算 CER / WER
  4. 结果写入 data/runs/ocr_eval/<ts>.json

在线 API 是异步批处理（上传 → 轮询 → 下载 zip），所以本脚本按 --batch-size
成批提交，而不是逐张阻塞。代价：latency 只能按批摊到单样本，不再是真实单张耗时，
结果 JSON 里用 latency_note 标注了这一点。

用法：
    uv run scripts/eval_ocr.py --limit 50
    uv run scripts/eval_ocr.py --limit 200 --model-version vlm
    uv run scripts/eval_ocr.py --limit 50 --doc-types academic_literature,research_report

前置：.env 里设置 MINERU_API_TOKEN（https://mineru.net/apiManage 创建）

依赖：
    pip install editdistance datasets pillow httpx tqdm
"""
from __future__ import annotations

import argparse
import io
import sys
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import PIL.Image
import editdistance
from dotenv import load_dotenv
from tqdm import tqdm

from lex_rag import ocr

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

    opts = ocr.options_from_args(args)
    print(f"MinerU 在线 API: {ocr.MINERU_API_BASE}  model_version={opts.model_version}  "
          f"is_ocr={opts.is_ocr}  batch_size={args.batch_size}")

    wall_t0 = time.perf_counter()
    with (
        ocr.MinerUCloudClient(options=opts) as client,
        tqdm(total=len(samples), desc="OCR 评测") as bar,
    ):
        for start_i in range(0, len(samples), args.batch_size):
            batch = samples[start_i:start_i + args.batch_size]

            # 只对当前批做 PNG 编码：全量 1651 张同时持有会吃掉数 GB 内存
            payload: list[tuple[str, bytes]] = []
            owners: list[int] = []          # owners[j] = payload[j] 属于 batch 里第几个样本
            for bi, item in enumerate(batch):
                images = split_columns(item["image"]) if args.column_split else [item["image"]]
                for ci, img in enumerate(images):
                    payload.append((f"s{start_i + bi}_c{ci}.png", pil_to_png_bytes(img)))
                    owners.append(bi)

            t0 = time.perf_counter()
            try:
                mds = client.parse(payload)
            except Exception as e:
                bar.write(f"  ⚠️ 批次 {start_i}~{start_i + len(batch)} 整批失败：{e}")
                bar.update(len(batch))
                continue
            per_item_ms = (time.perf_counter() - t0) * 1000 / max(len(batch), 1)

            # 列分割模式下一个样本对应多个文件，按 owners 归并回去
            merged: list[list[str]] = [[] for _ in batch]
            for md, owner in zip(mds, owners):
                merged[owner].append(md)

            for item, parts in zip(batch, merged):
                pred_md = "\n\n".join(p for p in parts if p)
                if not pred_md:
                    bar.write(f"  ⚠️ 空结果，跳过（{item['doc_type']}）")
                    continue
                pred_norm = normalize(pred_md)
                gt_norm   = normalize(item["gt_text"])
                row = {
                    "doc_type":   item["doc_type"],
                    "cer":        cer(pred_norm, gt_norm),
                    "wer":        wer(pred_norm, gt_norm),
                    "latency_ms": round(per_item_ms, 1),
                }
                all_results.append(row)
                results_by_type.setdefault(item["doc_type"], []).append(row)

            bar.update(len(batch))

    total_wall_sec = round(time.perf_counter() - wall_t0, 1)

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
        "api_base":         ocr.MINERU_API_BASE,
        "limit":            args.limit,
        "samples_per_type": args.samples_per_type,
        "doc_types":        list(doc_types),
        "lang":             opts.language,
        "model_version":    opts.model_version,
        "is_ocr":           opts.is_ocr,
        "enable_formula":   opts.enable_formula,
        "enable_table":     opts.enable_table,
        "batch_size":       args.batch_size,
        "column_split":     args.column_split,
        "total_wall_sec":   total_wall_sec,
        "latency_note":     "batch-amortized：单批墙钟时间 / 批内样本数，不是单张真实耗时",
        "overall":          overall,
        "by_type":          by_type,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",     type=int, default=50,
                        help="最多评测的样本数")
    parser.add_argument("--column-split", action="store_true",
                        help="启用列分割预处理：检测双栏布局并分列 OCR 后拼接，"
                             "针对 research_report 等多栏文档改善阅读顺序")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="每批提交给在线 API 的样本数（官方单批上限 200）。"
                             "批越大吞吐越高，但单批失败影响的样本也越多。")
    parser.add_argument("--samples-per-type", type=int, default=None,
                        help="每种文档类型固定取前 N 条样本（用于可复现的调优测试集）。"
                             "设置后 --limit 仍作为总量上限。")
    parser.add_argument("--doc-types", default=None,
                        help="逗号分隔的 data_source 值，默认全量。"
                             "可选: academic_literature,research_report,book,PPT2PDF,magazine,exam_paper,newspaper,note")
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
    run_eval(args)


if __name__ == "__main__":
    main()
