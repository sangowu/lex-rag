"""Unit tests for chunking.py pure-logic chunk-splitting functions."""

from lex_rag.chunking import chunk_fixed, chunk_recursive, chunk_parent_child


def test_chunk_fixed_basic_split():
    text = "abcdefghij"  # 10 chars
    chunks = list(chunk_fixed("doc1", text, chunk_chars=4, overlap=1))

    assert [c.text for c in chunks] == ["abcd", "defg", "ghij", "j"]
    assert [c.chunk_id for c in chunks] == ["doc1#0", "doc1#1", "doc1#2", "doc1#3"]
    assert [(c.start, c.end) for c in chunks] == [(0, 4), (3, 7), (6, 10), (9, 10)]


def test_chunk_fixed_empty_text():
    assert list(chunk_fixed("doc1", "", chunk_chars=10, overlap=2)) == []


def test_chunk_recursive_splits_on_paragraph_when_over_limit():
    text = "AAAA\n\nBBBB"
    chunks = list(chunk_recursive("doc1", text, chunk_chars=5, overlap=1))

    assert len(chunks) == 2
    assert chunks[0].text == "AAAA"
    assert (chunks[0].start, chunks[0].end) == (0, 6)
    assert chunks[1].text == "ABBBB"
    assert (chunks[1].start, chunks[1].end) == (3, 8)


def test_chunk_recursive_single_paragraph_fits_in_one_chunk():
    text = "hello world"
    chunks = list(chunk_recursive("doc1", text, chunk_chars=50, overlap=5))

    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert (chunks[0].start, chunks[0].end) == (0, 11)


def test_chunk_parent_child_links_children_to_parent():
    text = "0123456789"
    parents, children = chunk_parent_child(
        "doc1", text, parent_chars=10, child_chars=5, overlap=2
    )

    assert len(parents) == 1
    assert parents[0].chunk_id == "doc1#p0"
    assert parents[0].text == text

    assert [c.chunk_id for c in children] == ["doc1#p0c0", "doc1#p0c1", "doc1#p0c2"]
    assert [c.text for c in children] == ["01234", "34567", "6789"]
    assert all(c.parent_chunk_id == "doc1#p0" for c in children)
