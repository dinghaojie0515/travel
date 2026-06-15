"""
重排序工具

使用 BGE-Reranker-Large 对召回结果重新打分，并通过「断崖检测」
自动截断：当相邻两条结果分差超过阈值时，丢弃后续低分结果。
"""
from __future__ import annotations

from typing import Optional
from FlagEmbedding import FlagReranker
from loguru import logger

from app.config import settings

_reranker: Optional[FlagReranker] = None


def get_reranker() -> FlagReranker:
    """懒加载 BGE-Reranker-Large 模型"""
    global _reranker
    if _reranker is None:
        logger.info(f"加载 Reranker 模型: {settings.BGE_RERANKER_MODEL}（首次加载较慢）...")
        _reranker = FlagReranker(settings.BGE_RERANKER_MODEL, use_fp16=False)  # CPU 环境关闭 fp16
        logger.info("Reranker 模型加载完成")
    return _reranker


def rerank(
    query: str,
    docs: list[dict],
    top_k: int = 6,
    cliff_threshold: float = 0.15,
) -> list[dict]:
    """
    对检索结果重新打分并截断。

    参数：
        query           - 原始查询
        docs            - hybrid_search 返回的文档列表
        top_k           - 重排后最多保留条数
        cliff_threshold - 断崖阈值：相邻分差超过此值即截断（0~1）

    返回：重排后的文档列表，每条增加 "rerank_score" 字段
    """
    if not docs:
        return []

    reranker = get_reranker()
    pairs = [[query, doc["content"]] for doc in docs]

    try:
        scores = reranker.compute_score(pairs, normalize=True)  # 归一化到 0~1
    except TypeError:
        # 部分旧版 FlagEmbedding 不支持 normalize 参数，回退到原始分数
        scores = reranker.compute_score(pairs)
        logger.debug("Reranker 不支持 normalize 参数，使用原始分数")

    if isinstance(scores, (int, float)):
        scores = [float(scores)]
    else:
        scores = [float(s) for s in scores]

    # 按 rerank_score 降序排列
    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    # 截取 top_k
    ranked = ranked[:top_k]

    # 断崖检测：相邻两条分差超过阈值时截断
    cutoff = len(ranked)
    for i in range(1, len(ranked)):
        gap = float(ranked[i - 1][1]) - float(ranked[i][1])
        if gap >= cliff_threshold:
            cutoff = i
            logger.info(
                f"断崖检测触发：第{i}条与第{i+1}条分差={gap:.3f} >= {cliff_threshold}，截断"
            )
            break

    result = []
    for doc, score in ranked[:cutoff]:
        item = dict(doc)
        item["rerank_score"] = round(float(score), 4)
        result.append(item)

    logger.info(f"重排序完成: {len(docs)} → {len(result)} 条")
    return result
