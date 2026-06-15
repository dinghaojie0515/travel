"""
检索 LangGraph 工作流

节点顺序：
  parallel_retrieve → merge_results → rerank_results

parallel_retrieve 内部并发执行三路召回：
  A. 原始 query 的混合检索（稠密+稀疏向量）
  B. HyDE 假设文档的混合检索
  C. Web 搜索（Tavily，配置 TAVILY_API_KEY 后自动启用）
"""
from __future__ import annotations

import asyncio
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from loguru import logger

from app.utils.retriever import hybrid_search
from app.utils.reranker import rerank
from app.utils.llm_client import generate_hyde_doc
from app.utils.web_searcher import web_search, web_search_enabled


# ============================================================
# State 定义
# ============================================================
class RetrievalState(TypedDict):
    query:          str
    top_k:          int
    content_type:   str | None
    region:         str | None
    raw_docs:       list[dict]   # 合并去重后的召回结果
    reranked_docs:  list[dict]   # 重排序后的最终结果
    _hyde_docs:     list[dict]   # HyDE 召回的临时结果，merge 后不再使用
    _web_docs:      list[dict]   # Web 搜索临时结果，merge 后不再使用


# ============================================================
# 节点函数
# ============================================================
async def node_parallel_retrieve(state: RetrievalState) -> dict:
    """
    并发执行最多三路召回：
      A. 原始 query 混合检索
      B. HyDE 假设文档混合检索
      C. Web 搜索（可选，配置 TAVILY_API_KEY 后启用）
    """
    query        = state["query"]
    top_k        = state.get("top_k", 20)
    content_type = state.get("content_type")
    region       = state.get("region")

    active_paths = ["原始向量", "HyDE向量"]
    if web_search_enabled():
        active_paths.append("Web搜索")
    logger.info(f"开始并发检索（{' + '.join(active_paths)}）: query='{query[:40]}'")

    # HyDE 生成（需要先拿到 hyde_doc 才能发起第二路检索）
    hyde_doc = await generate_hyde_doc(query)

    # 构建并发任务
    tasks = [
        hybrid_search(query,    top_k=top_k, content_type=content_type, region=region),
        hybrid_search(hyde_doc, top_k=top_k, content_type=content_type, region=region),
    ]
    if web_search_enabled():
        tasks.append(web_search(query))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    original_docs = results[0] if not isinstance(results[0], Exception) else []
    hyde_docs     = results[1] if not isinstance(results[1], Exception) else []
    web_docs      = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else []

    if isinstance(results[0], Exception):
        logger.warning(f"原始向量检索异常: {results[0]}")
    if isinstance(results[1], Exception):
        logger.warning(f"HyDE 向量检索异常: {results[1]}")
    if len(results) > 2 and isinstance(results[2], Exception):
        logger.warning(f"Web 搜索异常: {results[2]}")

    logger.info(
        f"召回完成 — 原始: {len(original_docs)} 条，"
        f"HyDE: {len(hyde_docs)} 条，"
        f"Web: {len(web_docs)} 条"
    )
    return {"raw_docs": original_docs, "_hyde_docs": hyde_docs, "_web_docs": web_docs}


def node_merge_results(state: RetrievalState) -> dict:
    """
    合并三路召回结果，按 id 去重，保留最高分，按 score 降序排列。
    """
    original_docs = state.get("raw_docs",   [])
    hyde_docs     = state.get("_hyde_docs", [])
    web_docs      = state.get("_web_docs",  [])

    all_docs = original_docs + hyde_docs + web_docs
    seen: dict[str, dict] = {}
    for doc in all_docs:
        doc_id = doc["metadata"].get("id") or doc["content"][:48]
        if doc_id not in seen or doc["score"] > seen[doc_id]["score"]:
            seen[doc_id] = doc

    merged = sorted(seen.values(), key=lambda d: d["score"], reverse=True)
    web_count = sum(1 for d in merged if d["metadata"].get("content_type") == "web")
    logger.info(f"合并去重后: {len(merged)} 条（含 Web {web_count} 条）")
    return {"raw_docs": merged}


def node_rerank_results(state: RetrievalState) -> dict:
    """
    使用 BGE-Reranker-Large 对合并结果重排序，并做断崖截断。
    """
    reranked = rerank(
        query=state["query"],
        docs=state["raw_docs"],
        top_k=state.get("top_k", 6),
    )
    return {"reranked_docs": reranked}


# ============================================================
# 构建 Graph
# ============================================================
def build_retrieval_graph():
    graph = StateGraph(RetrievalState)

    graph.add_node("parallel_retrieve", node_parallel_retrieve)
    graph.add_node("merge_results",     node_merge_results)
    graph.add_node("rerank_results",    node_rerank_results)

    graph.add_edge(START,               "parallel_retrieve")
    graph.add_edge("parallel_retrieve", "merge_results")
    graph.add_edge("merge_results",     "rerank_results")
    graph.add_edge("rerank_results",    END)

    return graph.compile()


retrieval_graph = build_retrieval_graph()


async def retrieve(
    query: str,
    top_k: int = 6,
    content_type: str | None = None,
    region: str | None = None,
) -> list[dict]:
    """
    对外暴露的统一检索入口，供 RAG 问答 Graph 和搜索 API 调用。
    """
    result = await retrieval_graph.ainvoke({
        "query":        query,
        "top_k":        top_k,
        "content_type": content_type,
        "region":       region,
        "raw_docs":     [],
        "reranked_docs":[],
        "_hyde_docs":   [],
        "_web_docs":    [],
    })
    return result["reranked_docs"]
