"""
MinerU 在线 API（v4）客户端 —— 替代自建的 mineru-api 服务。

与原先本地 `POST /file_parse` 的关键差异（决定了本模块的形状）：

  本地：一次 multipart POST 直接拿到 `md_content`，同步。
  在线：三段式异步 ——
        ① POST /file-urls/batch  申请预签名上传链接（拿到 batch_id + file_urls）
        ② PUT  file_urls[i]      直传文件到 OSS（**不带 Authorization、不设 Content-Type**）
        ③ GET  /extract-results/batch/{batch_id}  轮询到 state=done
        ④ GET  full_zip_url      下载 zip，取出 full.md

因此本模块以 **批** 为单位工作：单文件也走批接口，避免维护两套代码路径。

用法::

    with MinerUCloudClient() as client:            # token 取自 MINERU_API_TOKEN
        md_list = client.parse([("a.png", png_bytes), ("b.pdf", pdf_bytes)])
        md = client.parse_one("c.png", png_bytes)

官方限制（截至 2026-08）：单文件 ≤200MB / ≤200 页，单批 ≤200 个文件，
上传链接有效期 24 小时；每账号每天 1000 页最高优先级额度，超出部分只降优先级、不拒绝。
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

MINERU_API_BASE = "https://mineru.net/api/v4"
TOKEN_ENV = "MINERU_API_TOKEN"

MAX_BATCH_FILES = 200          # 官方上限：单批 ≤200 个文件
_TERMINAL_STATES = {"done", "failed"}


class MinerUError(RuntimeError):
    """MinerU API 返回非 0 code、或流程中出现不可恢复错误。"""


@dataclass
class OcrOptions:
    """MinerU v4 的解析参数。

    注意与本地服务的参数不是一一对应：本地的 ``backend=hybrid-auto-engine`` /
    ``parse_method=ocr`` 在线上没有对应项，线上只有 ``model_version`` + ``is_ocr``。
    所以本地服务时期的 OmniDocBench 基线数字与线上结果不可直接比较，需要重跑。
    """
    model_version: str = "pipeline"   # pipeline | vlm | MinerU-HTML
    is_ocr: bool = True               # 强制走 OCR（扫描件必须为 True）
    enable_formula: bool = False
    enable_table: bool = True
    language: str = "ch"


class MinerUCloudClient:
    def __init__(
        self,
        token: str | None = None,
        base_url: str = MINERU_API_BASE,
        options: OcrOptions | None = None,
        timeout: float = 120.0,
        poll_interval: float = 5.0,
        poll_timeout: float = 1800.0,
        max_retries: int = 3,
        retry_backoff_sec: float = 2.0,
    ) -> None:
        self.token = token or os.environ.get(TOKEN_ENV, "")
        if not self.token:
            raise MinerUError(
                f"缺少 MinerU API token：请在 .env 里设置 {TOKEN_ENV}（在 "
                "https://mineru.net/apiManage 创建），或显式传入 token 参数。"
            )
        self.base_url = base_url.rstrip("/")
        self.options = options or OcrOptions()
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self._client = httpx.Client(timeout=httpx.Timeout(timeout), follow_redirects=True)

    # ── 生命周期 ────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MinerUCloudClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── 对外入口 ────────────────────────────────────────────────

    def parse(
        self,
        files: Sequence[tuple[str, bytes]],
        progress: Callable[[int], None] | None = None,
    ) -> list[str]:
        """解析一组 (文件名, 字节) ，返回**与输入顺序一致**的 Markdown 列表。

        单个文件解析失败不会中断整批：该位置返回空串，错误打印到 stderr 侧的日志行。
        超过单批上限时自动切分成多批顺序执行。
        ``progress`` 每完成一批被调用一次，参数为该批的文件数。
        """
        out: list[str] = []
        for start in range(0, len(files), MAX_BATCH_FILES):
            group = list(files[start:start + MAX_BATCH_FILES])
            out.extend(self._parse_group(group))
            if progress:
                progress(len(group))
        return out

    def parse_one(self, name: str, data: bytes) -> str:
        return self.parse([(name, data)])[0]

    def parse_paths(
        self,
        paths: Sequence[Path],
        progress: Callable[[int], None] | None = None,
    ) -> list[str]:
        """按路径读取文件后解析，返回与 paths 顺序一致的 Markdown 列表。"""
        files = [(p.name, p.read_bytes()) for p in paths]
        return self.parse(files, progress=progress)

    # ── 三段式流程 ──────────────────────────────────────────────

    def _parse_group(self, group: list[tuple[str, bytes]]) -> list[str]:
        # data_id 只需批内唯一；用序号即可，回填结果时靠它对齐顺序
        data_ids = [f"item-{i}" for i in range(len(group))]
        opts = self.options

        payload = {
            "files": [
                {"name": name, "data_id": did, "is_ocr": opts.is_ocr}
                for did, (name, _) in zip(data_ids, group)
            ],
            "model_version": opts.model_version,
            "enable_formula": opts.enable_formula,
            "enable_table": opts.enable_table,
            "language": opts.language,
        }
        data = self._post_json("/file-urls/batch", payload)
        batch_id = data["batch_id"]
        file_urls = data["file_urls"]
        if len(file_urls) != len(group):
            raise MinerUError(
                f"file_urls 数量({len(file_urls)})与请求文件数({len(group)})不一致，无法安全对齐"
            )

        for url, (name, blob) in zip(file_urls, group):
            self._upload(url, blob, name)

        zip_urls = self._poll_batch(batch_id, data_ids)
        return [self._fetch_markdown(zip_urls[did]) if zip_urls.get(did) else "" for did in data_ids]

    def _upload(self, presigned_url: str, blob: bytes, name: str) -> None:
        """PUT 直传到 OSS 预签名链接。

        官方明确要求 **不要设置 Content-Type**（会导致签名不匹配），
        也不要带 Authorization —— 签名已经在 URL 里。
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.put(presigned_url, content=blob)
                resp.raise_for_status()
                return
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_sec * (attempt + 1))
        raise MinerUError(f"上传 {name} 失败：{last_error}") from last_error

    def _poll_batch(self, batch_id: str, data_ids: list[str]) -> dict[str, str]:
        """轮询批次直到所有文件到达终态，返回 {data_id: full_zip_url}（失败项不入表）。"""
        pending = set(data_ids)
        zip_urls: dict[str, str] = {}
        deadline = time.monotonic() + self.poll_timeout

        while pending and time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            data = self._get_json(f"/extract-results/batch/{batch_id}")
            for item in data.get("extract_result", []):
                did = item.get("data_id")
                if did not in pending:
                    continue
                state = item.get("state")
                if state not in _TERMINAL_STATES:
                    continue
                pending.discard(did)
                if state == "done" and item.get("full_zip_url"):
                    zip_urls[did] = item["full_zip_url"]
                else:
                    print(f"[mineru] {item.get('file_name', did)} 解析失败："
                          f"state={state} err={item.get('err_msg', '')}", flush=True)

        if pending:
            print(f"[mineru] batch {batch_id} 有 {len(pending)} 个文件在 "
                  f"{self.poll_timeout:.0f}s 内未完成，按失败处理", flush=True)
        return zip_urls

    def _fetch_markdown(self, zip_url: str) -> str:
        """下载结果 zip 并取出 Markdown（优先 full.md，否则包内第一个 .md）。"""
        resp = self._client.get(zip_url)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            target = next((n for n in names if n.endswith("full.md")), None)
            if target is None:
                target = next((n for n in names if n.endswith(".md")), None)
            if target is None:
                print(f"[mineru] 结果包内没有 .md 文件：{names[:5]}", flush=True)
                return ""
            return zf.read(target).decode("utf-8", errors="replace")

    # ── HTTP 封装 ───────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        }

    def _post_json(self, path: str, payload: dict) -> dict:
        return self._request("POST", path, json=payload)

    def _get_json(self, path: str) -> dict:
        return self._request("GET", path)

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.request(method, url, headers=self._headers(), **kwargs)
                # 4xx 里只有 429 值得重试，其余（401/400）重试也不会变好，直接抛
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}",
                        request=resp.request, response=resp,
                    )
                resp.raise_for_status()
                body = resp.json()
                if body.get("code") not in (0, None):
                    raise MinerUError(
                        f"{method} {path} 返回 code={body.get('code')} "
                        f"msg={body.get('msg')} trace_id={body.get('trace_id')}"
                    )
                return body.get("data", {})
            except MinerUError:
                raise
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_sec * (attempt + 1))
        raise MinerUError(f"{method} {path} 重试 {self.max_retries} 次后仍失败：{last_error}") from last_error


def options_from_args(args) -> OcrOptions:
    """从 argparse 结果构造 OcrOptions（三个 OCR 脚本共用同一组参数名）。"""
    return OcrOptions(
        model_version=args.model_version,
        is_ocr=not getattr(args, "no_ocr", False),
        enable_formula=getattr(args, "formula", False),
        enable_table=not getattr(args, "no_table", False),
        language=args.lang,
    )


def add_ocr_args(parser) -> None:
    """把 MinerU 在线 API 的公共参数挂到 argparse parser 上。"""
    parser.add_argument("--model-version", default="pipeline",
                        choices=["pipeline", "vlm", "MinerU-HTML"],
                        help="MinerU 线上模型版本（默认 pipeline）")
    parser.add_argument("--lang", default="ch",
                        help="OCR 语言，默认 ch")
    parser.add_argument("--no-ocr", action="store_true",
                        help="关闭强制 OCR（有文字层的 PDF 可加速；扫描件不要加）")
    parser.add_argument("--formula", action="store_true",
                        help="开启公式识别（默认关闭，与本地基线配置一致）")
    parser.add_argument("--no-table", action="store_true",
                        help="关闭表格识别（默认开启）")
