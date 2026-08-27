from collections import defaultdict
import contextlib
import json
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from lex_rag.chunking import ChunkWindow


class VectorStore:
    def __init__(self, dsn: str, table: str = "chunks", init_schema: bool = True):
        self.conn = psycopg.connect(dsn)
        self.table = table
        with self._cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.commit()
        register_vector(self.conn)
        if init_schema:
            self._init_schema()

    @contextlib.contextmanager
    def _cursor(self):
        """取游标，**出错时回滚**。

        psycopg 在事务中遇到任何错误后，该连接上的后续语句一律报
        `InFailedSqlTransaction`，直到显式 rollback。不回滚的话，一次瞬时错误
        （并发 DDL 死锁、语句超时、网络抖动）会把连接**永久**报废——而 `serve.py`
        是长驻进程，症状会是"从此所有查询都失败"，而不是"那一次失败"，排查时
        极容易看错方向。

        实测：4 个 worker 同时构造 VectorStore 触发 DDL 死锁后，同一批 20 条里
        15 条全部报 InFailedSqlTransaction，真正的死锁只发生过一次。
        """
        try:
            with self.conn.cursor() as cur:
                yield cur
        except Exception:
            with contextlib.suppress(Exception):
                self.conn.rollback()
            raise

    def _init_schema(self) -> None:
        t = self.table
        with self._cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {t} (
                    chunk_id        TEXT PRIMARY KEY,
                    doc_id          TEXT NOT NULL,
                    text            TEXT NOT NULL,
                    start_pos       INT,
                    end_pos         INT,
                    embedding       vector(1024),
                    tsv             tsvector GENERATED ALWAYS AS (
                                        to_tsvector('english', text)
                                    ) STORED,
                    parent_chunk_id TEXT
                )
            """)
            # 兼容旧表：若 parent_chunk_id 列不存在则添加
            cur.execute(f"""
                ALTER TABLE {t} ADD COLUMN IF NOT EXISTS parent_chunk_id TEXT
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {t}_embedding_idx
                ON {t} USING hnsw (embedding vector_cosine_ops)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {t}_tsv_idx
                ON {t} USING GIN (tsv)
            """)
            # ingest 参数元数据表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ingest_meta (
                    table_name   TEXT PRIMARY KEY,
                    chunk_chars  INT,
                    overlap      INT,
                    strategy     TEXT,
                    contextual   BOOLEAN,
                    chunk_mode   TEXT DEFAULT 'standard',
                    ingested_at  TIMESTAMPTZ DEFAULT now()
                )
            """)
            cur.execute("""
                ALTER TABLE ingest_meta ADD COLUMN IF NOT EXISTS chunk_mode TEXT DEFAULT 'standard'
            """)
            # 文档级 metadata 表（所有表共用一张）
            cur.execute("""
                CREATE TABLE IF NOT EXISTS doc_meta (
                    doc_id         TEXT PRIMARY KEY,
                    contract_type  TEXT,
                    party_a        TEXT,
                    party_b        TEXT,
                    effective_date TEXT,
                    governing_law  TEXT,
                    key_clauses    TEXT[],
                    raw_json       JSONB,
                    extracted_at   TIMESTAMPTZ DEFAULT now()
                )
            """)
        self.conn.commit()

    def save_meta(self, chunk_chars: int, overlap: int, strategy: str,
                  contextual: bool, chunk_mode: str = "standard") -> None:
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO ingest_meta
                    (table_name, chunk_chars, overlap, strategy, contextual, chunk_mode, ingested_at)
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (table_name) DO UPDATE
                    SET chunk_chars = EXCLUDED.chunk_chars,
                        overlap     = EXCLUDED.overlap,
                        strategy    = EXCLUDED.strategy,
                        contextual  = EXCLUDED.contextual,
                        chunk_mode  = EXCLUDED.chunk_mode,
                        ingested_at = EXCLUDED.ingested_at
            """, (self.table, chunk_chars, overlap, strategy, contextual, chunk_mode))
        self.conn.commit()

    def load_meta(self) -> dict | None:
        with self._cursor() as cur:
            cur.execute("""
                SELECT chunk_chars, overlap, strategy, contextual, chunk_mode, ingested_at
                FROM ingest_meta WHERE table_name = %s
            """, (self.table,))
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "chunk_chars": row[0],
            "overlap":     row[1],
            "strategy":    row[2],
            "contextual":  row[3],
            "chunk_mode":  row[4] or "standard",
            "ingested_at": row[5].isoformat() if row[5] else None,
        }

    def truncate(self) -> None:
        with self._cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self.table}")
        self.conn.commit()

    def add_chunks(self, chunks: list[ChunkWindow], embeddings: list[list[float]]) -> None:
        with self._cursor() as cur:
            for chunk, embedding in zip(chunks, embeddings):
                cur.execute(f"""
                    INSERT INTO {self.table}
                        (chunk_id, doc_id, text, start_pos, end_pos, embedding, parent_chunk_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO NOTHING
                """, (chunk.chunk_id, chunk.doc_id, chunk.text, chunk.start, chunk.end,
                      embedding, chunk.parent_chunk_id))
        self.conn.commit()

    def expand_to_parent(self, children: list[ChunkWindow]) -> list[ChunkWindow]:
        """
        给定 child ChunkWindow 列表，返回去重后的 parent ChunkWindow 列表（按首次出现顺序）。
        若 child 无 parent（parent_chunk_id IS NULL），原样保留该 child。
        """
        if not children:
            return []
        child_ids = [c.chunk_id for c in children]
        with self._cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id, parent_chunk_id
                FROM {self.table}
                WHERE chunk_id = ANY(%s)
            """, (child_ids,))
            id_to_parent = {row[0]: row[1] for row in cur.fetchall()}

        seen_parent_ids: list[str] = []
        seen_set: set[str] = set()
        no_parent: list[ChunkWindow] = []
        # parent 继承命中它的 child 里最高的那个分数——parent 自己没有被检索过，
        # 不继承的话 trace 里这一段就是一片 None，没法判断召回质量。
        parent_score: dict[str, tuple[float, str | None]] = {}
        for child in children:
            pid = id_to_parent.get(child.chunk_id)
            if pid and child.score is not None:
                prev = parent_score.get(pid)
                if prev is None or child.score > prev[0]:
                    parent_score[pid] = (child.score, child.score_kind)
            if pid and pid not in seen_set:
                seen_parent_ids.append(pid)
                seen_set.add(pid)
            elif not pid:
                no_parent.append(child)

        parent_map: dict[str, ChunkWindow] = {}
        if seen_parent_ids:
            with self._cursor() as cur:
                cur.execute(f"""
                    SELECT chunk_id, doc_id, text, start_pos, end_pos
                    FROM {self.table}
                    WHERE chunk_id = ANY(%s)
                """, (seen_parent_ids,))
                for row in cur.fetchall():
                    sc, kind = parent_score.get(row[0], (None, None))
                    parent_map[row[0]] = ChunkWindow(
                        chunk_id=row[0], doc_id=row[1], text=row[2],
                        start=row[3], end=row[4], score=sc, score_kind=kind,
                    )

        result = [parent_map[pid] for pid in seen_parent_ids if pid in parent_map]
        result.extend(no_parent)
        return result

    def add_doc_meta(self, doc_id: str, meta: dict) -> None:
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO doc_meta
                    (doc_id, contract_type, party_a, party_b, effective_date,
                     governing_law, key_clauses, raw_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE
                    SET contract_type  = EXCLUDED.contract_type,
                        party_a        = EXCLUDED.party_a,
                        party_b        = EXCLUDED.party_b,
                        effective_date = EXCLUDED.effective_date,
                        governing_law  = EXCLUDED.governing_law,
                        key_clauses    = EXCLUDED.key_clauses,
                        raw_json       = EXCLUDED.raw_json,
                        extracted_at   = now()
            """, (
                doc_id,
                meta.get("contract_type"),
                meta.get("party_a"),
                meta.get("party_b"),
                meta.get("effective_date"),
                meta.get("governing_law"),
                meta.get("key_clauses", []),
                json.dumps(meta),
            ))
        self.conn.commit()

    def get_doc_meta(self, doc_id: str) -> dict | None:
        with self._cursor() as cur:
            cur.execute("""
                SELECT contract_type, party_a, party_b, effective_date,
                       governing_law, key_clauses, raw_json
                FROM doc_meta WHERE doc_id = %s
            """, (doc_id,))
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "contract_type": row[0], "party_a": row[1], "party_b": row[2],
            "effective_date": row[3], "governing_law": row[4],
            "key_clauses": row[5], "raw_json": row[6],
        }

    def search_vector(self, query_vec: list[float], k: int,
                      doc_id: str | None = None,
                      children_only: bool = False) -> list[ChunkWindow]:
        vec = np.array(query_vec)
        child_filter = "AND parent_chunk_id IS NOT NULL" if children_only else ""
        with self._cursor() as cur:
            # 距离进 SELECT 列表并按别名排序：ORDER BY 引用输出别名时 Postgres 会
            # 解析回原表达式，pgvector 索引照常命中，但分数拿得到了。
            if doc_id is not None:
                cur.execute(f"""
                    SELECT chunk_id, doc_id, text, start_pos, end_pos,
                           embedding <=> %s AS dist
                    FROM {self.table}
                    WHERE doc_id = %s {child_filter}
                    ORDER BY dist
                    LIMIT %s
                """, (vec, doc_id, k))
            else:
                cur.execute(f"""
                    SELECT chunk_id, doc_id, text, start_pos, end_pos,
                           embedding <=> %s AS dist
                    FROM {self.table}
                    WHERE TRUE {child_filter}
                    ORDER BY dist
                    LIMIT %s
                """, (vec, k))
            rows = cur.fetchall()
        # 距离转相似度：让所有 score 统一为"越大越相关"
        return [ChunkWindow(chunk_id=r[0], doc_id=r[1], text=r[2], start=r[3], end=r[4],
                            score=1.0 - float(r[5]), score_kind="cosine_sim") for r in rows]

    def search_bm25(self, query: str, k: int,
                    doc_id: str | None = None,
                    children_only: bool = False) -> list[ChunkWindow]:
        child_filter = "AND parent_chunk_id IS NOT NULL" if children_only else ""
        with self._cursor() as cur:
            cur.execute("SELECT replace(plainto_tsquery('english', %s)::text, ' & ', ' | ')", (query,))
            tsq_or = cur.fetchone()[0]
            if not tsq_or:
                return []
            if doc_id is not None:
                cur.execute(f"""
                    SELECT chunk_id, doc_id, text, start_pos, end_pos,
                           ts_rank_cd(tsv, to_tsquery('english', %s)) AS rank
                    FROM {self.table}
                    WHERE doc_id = %s
                      AND tsv @@ to_tsquery('english', %s)
                      {child_filter}
                    ORDER BY rank DESC
                    LIMIT %s
                """, (tsq_or, doc_id, tsq_or, k))
            else:
                cur.execute(f"""
                    SELECT chunk_id, doc_id, text, start_pos, end_pos,
                           ts_rank_cd(tsv, to_tsquery('english', %s)) AS rank
                    FROM {self.table}
                    WHERE tsv @@ to_tsquery('english', %s)
                      {child_filter}
                    ORDER BY rank DESC
                    LIMIT %s
                """, (tsq_or, tsq_or, k))
            rows = cur.fetchall()
        return [ChunkWindow(chunk_id=r[0], doc_id=r[1], text=r[2], start=r[3], end=r[4],
                            score=float(r[5]), score_kind="bm25") for r in rows]

    def search_hybrid(self, query: str, query_vec: list[float], k: int,
                      doc_id: str | None = None,
                      children_only: bool = False) -> list[ChunkWindow]:
        vec_results  = self.search_vector(query_vec, k, doc_id, children_only=children_only)
        bm25_results = self.search_bm25(query, k, doc_id, children_only=children_only)
        scores: dict[str, float] = defaultdict(float)
        for rank, chunk in enumerate(vec_results):
            scores[chunk.chunk_id] += 1 / (rank + 60)
        for rank, chunk in enumerate(bm25_results):
            scores[chunk.chunk_id] += 1 / (rank + 60)
        chunk_map = {c.chunk_id: c for c in vec_results + bm25_results}
        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        out = []
        for cid in sorted_ids[:k]:
            chunk = chunk_map[cid]
            # 覆盖掉单路的 cosine/bm25 分数：返回的是融合排名，分数也该是融合分，
            # 否则 trace 里的分数与它自己的名次对不上。
            chunk.score, chunk.score_kind = scores[cid], "rrf"
            out.append(chunk)
        return out

    def close(self) -> None:
        self.conn.close()
