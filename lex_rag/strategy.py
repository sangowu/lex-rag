"""
RetrievalStrategy —— 把"这一轮检索怎么做"从配置文件里搬到运行时。

改造前，五个检索技术（hybrid/vector/bm25、HyDE、multi-query、parent 展开、
rerank）都是在 `RAGPipeline.__init__` 里由 config 决定的，一旦构造完就改不了。
这意味着"策略选择"根本不存在：同一个 pipeline 对所有问题做同一件事。

法律合同里既有"违约金是多少"这种精确术语查询（BM25 更强），也有"这份合同对
乙方有哪些限制"这种概念性查询（向量更强）。静态配置只能取一个折中值。

本对象把这些开关变成一次调用的参数。**它是纯数据、不可变**，因此可以被记录进
trace、被比较、被去重——这三点是后续做失败归因的前提。

向后兼容：`_query_impl(strategy=None)` 时用 `from_config()` 构造出与改造前完全
等价的默认策略，所以现有调用方一行都不用改。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

Mode = Literal["vector", "bm25", "hybrid"]


@dataclass(frozen=True)
class RetrievalStrategy:
    """一轮检索的完整描述。frozen 是有意的——见模块 docstring。"""

    mode: Mode = "hybrid"
    use_hyde: bool = False
    use_multi_query: bool = False
    multi_query_n: int = 3          # 含原始问题在内的总查询数
    fetch_k: int = 60               # 送进 reranker 的候选数
    top_k: int = 10                 # 最终返回条数
    expand_parent: bool = True      # 仅在表本身是 parent_child 模式时有意义
    rerank: bool = True
    query_text: str | None = None   # 重写后的查询；None 表示用原问题

    @classmethod
    def from_config(cls, cfg) -> "RetrievalStrategy":
        """从 AppConfig 构造默认策略，行为与改造前的 `_query_impl` 完全一致。

        注意 fetch_k 的取法：改造前是 `rerank_top_k if reranker else k`，
        也就是不开 rerank 时候选数就等于 top_k。这里保持同样的逻辑，否则
        回归验证会出现无法解释的差异。
        """
        rerank_on = cfg.reranker.enabled
        return cls(
            mode=cfg.retrieval.mode,
            use_hyde=cfg.hyde_enabled,
            use_multi_query=cfg.multi_query_enabled,
            multi_query_n=cfg.multi_query_n,
            fetch_k=cfg.retrieval.rerank_top_k if rerank_on else cfg.retrieval.top_k,
            top_k=cfg.retrieval.top_k,
            expand_parent=True,
            rerank=rerank_on,
        )

    def with_top_k(self, k: int | None) -> "RetrievalStrategy":
        """调用方显式传了 k 时覆盖。

        不开 rerank 时 fetch_k 要跟着走，否则候选池会与 top_k 脱节——
        这正是改造前 `fetch_k = rerank_top_k if reranker else k` 的语义。
        """
        if k is None or k == self.top_k:
            return self
        if self.rerank:
            return replace(self, top_k=k)
        return replace(self, top_k=k, fetch_k=k)

    def key(self) -> str:
        """策略指纹，用于防重复：同一策略不允许在一次运行里跑两次。

        包含 query_text 的哈希而不是全文——重写后的查询可能很长，而这里只需要
        回答"这两轮是不是同一件事"。
        """
        import hashlib

        qt = "none"
        if self.query_text is not None:
            qt = hashlib.sha256(self.query_text.encode("utf-8")).hexdigest()[:8]
        return (
            f"{self.mode}|hyde={int(self.use_hyde)}|mq={int(self.use_multi_query)}"
            f":{self.multi_query_n if self.use_multi_query else 0}"
            f"|fetch={self.fetch_k}|top={self.top_k}"
            f"|parent={int(self.expand_parent)}|rerank={int(self.rerank)}|q={qt}"
        )

    def to_dict(self) -> dict:
        """落盘用。trace 里要能还原出"系统当时选了什么"。"""
        return {
            "mode": self.mode,
            "use_hyde": self.use_hyde,
            "use_multi_query": self.use_multi_query,
            "multi_query_n": self.multi_query_n,
            "fetch_k": self.fetch_k,
            "top_k": self.top_k,
            "expand_parent": self.expand_parent,
            "rerank": self.rerank,
            "query_text": self.query_text,
            "key": self.key(),
        }
