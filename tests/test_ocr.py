"""MinerU 在线 API 客户端的单元测试（全部网络调用 mock 掉）。

覆盖三段式流程里最容易出错、且一旦出错就会静默给出错误结果的两点：
  1. 结果按 data_id 回填 —— 轮询返回的顺序与提交顺序无关，错位会让
     每个样本对上别人的 ground truth，CER 依然算得出来，但整份评测是废的；
  2. zip 里取 full.md —— 取错文件（如 layout.json）同样不会报错。
"""

import io
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from lex_rag.ocr import MinerUCloudClient, MinerUError, OcrOptions


def _zip_bytes(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _client(**kwargs) -> MinerUCloudClient:
    with patch("httpx.Client"):
        return MinerUCloudClient(
            token="fake-token",
            options=OcrOptions(),
            poll_interval=0.0,
            max_retries=0,
            retry_backoff_sec=0.0,
            **kwargs,
        )


def test_missing_token_raises_with_env_var_name():
    with patch.dict("os.environ", {}, clear=True), patch("httpx.Client"):
        with pytest.raises(MinerUError, match="MINERU_API_TOKEN"):
            MinerUCloudClient()


def test_nonzero_code_in_envelope_raises():
    client = _client()
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"code": -60001, "msg": "quota exceeded", "trace_id": "t1"}
    client._client.request.return_value = resp

    with pytest.raises(MinerUError, match="quota exceeded"):
        client._get_json("/whatever")


def test_results_are_mapped_back_by_data_id_not_by_response_order():
    """轮询结果乱序返回时，Markdown 仍必须按提交顺序对齐。"""
    client = _client()
    client._post_json = MagicMock(return_value={
        "batch_id": "b1",
        "file_urls": ["https://upload/0", "https://upload/1"],
    })
    client._upload = MagicMock()
    # 故意把 item-1 放在前面：真实 API 不保证顺序
    client._get_json = MagicMock(return_value={"extract_result": [
        {"data_id": "item-1", "state": "done", "full_zip_url": "https://zip/1"},
        {"data_id": "item-0", "state": "done", "full_zip_url": "https://zip/0"},
    ]})
    client._fetch_markdown = MagicMock(side_effect=lambda url: f"md-from-{url[-1]}")

    out = client.parse([("a.png", b"A"), ("b.png", b"B")])

    assert out == ["md-from-0", "md-from-1"]


def test_failed_item_yields_empty_string_without_aborting_the_batch():
    client = _client()
    client._post_json = MagicMock(return_value={
        "batch_id": "b1",
        "file_urls": ["https://upload/0", "https://upload/1"],
    })
    client._upload = MagicMock()
    client._get_json = MagicMock(return_value={"extract_result": [
        {"data_id": "item-0", "state": "failed", "err_msg": "unsupported file"},
        {"data_id": "item-1", "state": "done", "full_zip_url": "https://zip/1"},
    ]})
    client._fetch_markdown = MagicMock(return_value="ok-md")

    out = client.parse([("a.png", b"A"), ("b.png", b"B")])

    assert out == ["", "ok-md"]


def test_parse_splits_into_groups_of_at_most_200():
    client = _client()
    seen_group_sizes = []
    client._parse_group = MagicMock(
        side_effect=lambda g: (seen_group_sizes.append(len(g)), [""] * len(g))[1]
    )

    client.parse([(f"{i}.png", b"x") for i in range(450)])

    assert seen_group_sizes == [200, 200, 50]


def test_fetch_markdown_prefers_full_md():
    client = _client()
    resp = MagicMock(status_code=200)
    resp.content = _zip_bytes({
        "layout.json": '{"not": "markdown"}',
        "full.md": "# real content",
    })
    client._client.get.return_value = resp

    assert client._fetch_markdown("https://zip/x") == "# real content"


def test_fetch_markdown_returns_empty_when_zip_has_no_markdown():
    client = _client()
    resp = MagicMock(status_code=200)
    resp.content = _zip_bytes({"layout.json": "{}"})
    client._client.get.return_value = resp

    assert client._fetch_markdown("https://zip/x") == ""


def test_upload_sends_raw_bytes_without_auth_or_content_type():
    """预签名 URL 自带签名：多带 Content-Type 或 Authorization 会导致签名不匹配。"""
    client = _client()
    resp = MagicMock(status_code=200)
    client._client.put.return_value = resp

    client._upload("https://upload/0", b"PNGDATA", "a.png")

    _, kwargs = client._client.put.call_args
    assert kwargs == {"content": b"PNGDATA"}


def test_poll_gives_up_after_timeout_and_reports_missing_items():
    client = _client(poll_timeout=0.05)
    client._get_json = MagicMock(return_value={"extract_result": [
        {"data_id": "item-0", "state": "running"},
    ]})

    assert client._poll_batch("b1", ["item-0"]) == {}
