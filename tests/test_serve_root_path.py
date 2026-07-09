"""Unit tests for RootPathMiddleware (subpath/ALB path-routing support).

ALB path-based routing forwards the full unmodified path to the target
(no prefix stripping), and uvicorn's own root_path option only affects URL
generation, not routing -- so this middleware manually strips the prefix.
"""

import asyncio

from scripts.serve import RootPathMiddleware


def _run_request(root_path: str, path: str) -> dict:
    captured: dict = {}

    async def fake_app(scope, receive, send):
        captured["path"] = scope.get("path")
        captured["root_path"] = scope.get("root_path")

    async def receive():
        return {"type": "http.request"}

    async def send(_message):
        pass

    mw = RootPathMiddleware(fake_app, root_path)
    scope = {"type": "http", "path": path, "root_path": ""}
    asyncio.run(mw(scope, receive, send))
    return captured


def test_strips_matching_prefix_and_sets_root_path():
    captured = _run_request("/legal-rag", "/legal-rag/health")

    assert captured == {"path": "/health", "root_path": "/legal-rag"}


def test_bare_prefix_rewrites_to_root():
    captured = _run_request("/legal-rag", "/legal-rag")

    assert captured["path"] == "/"


def test_non_matching_path_passes_through_unchanged():
    captured = _run_request("/legal-rag", "/health")

    assert captured == {"path": "/health", "root_path": ""}


def test_empty_root_path_is_a_no_op():
    captured = _run_request("", "/legal-rag/health")

    assert captured == {"path": "/legal-rag/health", "root_path": ""}
