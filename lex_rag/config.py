from dataclasses import dataclass, field
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

@dataclass
class EmbeddingConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    batch_size: int
    max_retries: int
    retry_backoff_sec: float
    # SSH tunnel（provider=ssh_tunnel 时生效）
    ssh_host: str = ""
    ssh_user: str = ""
    ssh_port: int = 22
    ssh_key_path: str = ""
    ssh_local_port: int = 6006
    ssh_remote_host: str = "127.0.0.1"
    ssh_remote_port: int = 8000

@dataclass
class DatabaseConfig:
    host: str
    port: int
    name: str
    user: str
    password: str  # 从 .env 注入
    table: str = "chunks"

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

@dataclass
class ChunkingConfig:
    strategy: str
    chunk_chars: int
    overlap: int

@dataclass
class RetrievalConfig:
    mode: str
    top_k: int
    rerank_top_k: int = 20

@dataclass
class RerankConfig:
    enabled: bool
    provider: str
    model: str
    base_url: str
    api_key: str
    batch_size: int
    max_retries: int
    retry_backoff_sec: float

@dataclass
class EvalConfig:
    k_values: list[int]
    scope: str
    mode: str

@dataclass
class ContextualConfig:
    enabled: bool
    model: str
    api_key: str
    rpm_limit: int
    max_retries: int
    retry_backoff_sec: float
    section_chars: int = 3000  # HierarchicalContextualizer 的 section 切分大小

@dataclass
class ParentChildConfig:
    parent_chars: int = 1000
    child_chars: int = 300
    overlap: int = 50

@dataclass
class RagasConfig:
    model: str = "gemini-2.0-flash"
    api_key: str = ""
    rpm_limit: int = 60

@dataclass
class AppConfig:
    embedding: EmbeddingConfig
    database: DatabaseConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    reranker: RerankConfig
    evaluation: EvalConfig
    contextual: ContextualConfig
    parent_child: ParentChildConfig = field(default_factory=ParentChildConfig)
    ragas: RagasConfig = field(default_factory=RagasConfig)
    hyde_enabled: bool = False
    multi_query_enabled: bool = False
    multi_query_n: int = 3        # 含原始问题在内的总查询数
    chunk_mode: str = "standard"       # standard | parent_child
    contextual_mode: str = "standard"  # standard | hierarchical
    extract_meta: bool = False         # ingest 时是否提取文档 metadata 存入 doc_meta 表

def load_config(config_path: Path | None = None) -> AppConfig:
    load_dotenv()
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    with open(config_path, "r", encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    api_key = os.environ["EMBED_API_KEY"]
    pg_password = os.environ["PG_PASSWORD"]

    emb = config_dict["embedding"]
    embedding_config = EmbeddingConfig(
        provider=emb["provider"],
        model=emb["model"],
        base_url=emb["base_url"],
        api_key=api_key,
        batch_size=emb["batch_size"],
        max_retries=emb["max_retries"],
        retry_backoff_sec=emb["retry_backoff_sec"],
        ssh_host=emb.get("ssh_host", ""),
        ssh_user=emb.get("ssh_user", ""),
        ssh_port=emb.get("ssh_port", 22),
        ssh_key_path=emb.get("ssh_key_path", ""),
        ssh_local_port=emb.get("ssh_local_port", 6006),
        ssh_remote_host=emb.get("ssh_remote_host", "127.0.0.1"),
        ssh_remote_port=emb.get("ssh_remote_port", 8000),
    )

    db = config_dict["database"]
    database_config = DatabaseConfig(
        host=db["host"],
        port=db["port"],
        name=db["name"],
        user=db["user"],
        password=pg_password,
        table=db.get("table", "chunks"),
    )

    chunking_config = ChunkingConfig(
        strategy=config_dict["chunking"]["strategy"],
        chunk_chars=config_dict["chunking"]["chunk_chars"],   
        overlap=config_dict["chunking"]["overlap"]
    )

    retrieval_config = RetrievalConfig(
        mode=config_dict["retrieval"]["mode"],
        top_k=config_dict["retrieval"]["top_k"],
        rerank_top_k=config_dict["retrieval"].get("rerank_top_k", 20),
    )

    rr = config_dict.get("reranker", {})
    reranker_config = RerankConfig(
        enabled=rr.get("enabled", False),
        provider=rr.get("provider", "direct"),
        model=rr.get("model", "BAAI/bge-reranker-v2-m3"),
        base_url=rr.get("base_url", ""),
        api_key=os.environ.get("RERANK_API_KEY", os.environ.get("EMBED_API_KEY", "")),
        batch_size=rr.get("batch_size", 32),
        max_retries=rr.get("max_retries", 2),
        retry_backoff_sec=rr.get("retry_backoff_sec", 1.0),
    )

    eval_config = EvalConfig(
        k_values=config_dict["evaluation"]["k_values"],
        scope=config_dict["evaluation"]["scope"],
        mode=config_dict["evaluation"]["mode"]
    )

    ctx = config_dict.get("contextual", {})
    contextual_config = ContextualConfig(
        enabled=ctx.get("enabled", False),
        model=ctx.get("model", "gemini-2.0-flash"),
        api_key=os.environ.get("GEMINI_API_KEY", ""),
        rpm_limit=ctx.get("rpm_limit", 60),
        max_retries=ctx.get("max_retries", 3),
        retry_backoff_sec=ctx.get("retry_backoff_sec", 2.0),
        section_chars=ctx.get("section_chars", 3000),
    )

    pc = config_dict.get("parent_child", {})
    parent_child_config = ParentChildConfig(
        parent_chars=pc.get("parent_chars", 1000),
        child_chars=pc.get("child_chars", 300),
        overlap=pc.get("overlap", 50),
    )

    hyde_enabled = config_dict.get("hyde", {}).get("enabled", False)
    mq = config_dict.get("multi_query", {})
    multi_query_enabled = mq.get("enabled", False)
    multi_query_n = mq.get("n", 3)

    rg = config_dict.get("ragas", {})
    ragas_config = RagasConfig(
        model=rg.get("model", "gemini-2.0-flash"),
        api_key=os.environ.get("GEMINI_API_KEY", ""),
        rpm_limit=rg.get("rpm_limit", 60),
    )

    return AppConfig(
        embedding=embedding_config,
        database=database_config,
        chunking=chunking_config,
        retrieval=retrieval_config,
        reranker=reranker_config,
        evaluation=eval_config,
        contextual=contextual_config,
        parent_child=parent_child_config,
        ragas=ragas_config,
        hyde_enabled=hyde_enabled,
        multi_query_enabled=multi_query_enabled,
        multi_query_n=multi_query_n,
    )