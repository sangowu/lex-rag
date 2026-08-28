"""连接必须是 autocommit，写路径必须显式开事务。

钉的是一次真实事故：`psycopg.connect(dsn)` 默认 autocommit=False，`_cursor()`
只在出错时 rollback，于是**只读**查询做完连接就停在 `idle in transaction`，
一直攥着表锁。症状是另一个进程的 `ALTER TABLE` 无限挂死（没有报错，只是不动），
以及长驻的 serve.py 让 VACUUM 回收不掉死元组。

这两条都不会被普通功能测试发现——查询照样返回正确结果。所以在这里钉住。
"""

from unittest.mock import MagicMock

import pytest

from lex_rag import store as store_mod
from lex_rag.chunking import ChunkWindow
from lex_rag.store import VectorStore


@pytest.fixture
def fake_conn(monkeypatch):
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = MagicMock()
    captured = {}

    def fake_connect(dsn, **kw):
        captured["dsn"] = dsn
        captured["kwargs"] = kw
        return conn

    monkeypatch.setattr(store_mod.psycopg, "connect", fake_connect)
    monkeypatch.setattr(store_mod, "register_vector", lambda _c: None)
    return conn, captured


def test_connection_is_autocommit(fake_conn):
    _conn, captured = fake_conn
    VectorStore("dsn://x", init_schema=False)

    assert captured["kwargs"].get("autocommit") is True, (
        "默认 autocommit=False 会让只读查询把连接留在 idle in transaction"
    )


def test_read_path_never_commits_because_there_is_nothing_to_commit(fake_conn):
    """autocommit 之后，读路径不该再靠 commit()/rollback() 收尾。"""
    conn, _ = fake_conn
    store = VectorStore("dsn://x", init_schema=False)
    conn.commit.reset_mock()

    store.load_meta()

    conn.commit.assert_not_called()


def test_add_chunks_opens_an_explicit_transaction(fake_conn):
    """循环里一行一条 INSERT：不包事务就是每行一次 fsync，且中途崩会写一半。"""
    conn, _ = fake_conn
    store = VectorStore("dsn://x", init_schema=False)
    conn.transaction.reset_mock()

    store.add_chunks(
        [ChunkWindow(chunk_id="c1", doc_id="d", text="t", start=0, end=1)],
        [[0.1] * 4],
    )

    conn.transaction.assert_called_once()


def test_single_statement_writes_do_not_need_a_transaction(fake_conn):
    """单条语句在 autocommit 下自带原子性，包事务只会白白拉长锁窗口。"""
    conn, _ = fake_conn
    store = VectorStore("dsn://x", init_schema=False)
    conn.transaction.reset_mock()

    store.truncate()
    store.save_meta(1000, 100, "recursive", False)

    conn.transaction.assert_not_called()


def test_init_schema_does_not_wrap_ddl_in_one_transaction(fake_conn):
    """DDL 全是 IF NOT EXISTS，逐条提交能让每把锁尽早释放。

    包成一个事务会拉长锁窗口，而多 worker 并发建表互相死锁正是这里出过的事故。
    """
    conn, _ = fake_conn
    conn.transaction.reset_mock()

    VectorStore("dsn://x", init_schema=True)

    conn.transaction.assert_not_called()
