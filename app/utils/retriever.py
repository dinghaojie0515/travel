"""
混合检索工具

流程：
  1. 对查询生成稠密向量（text-embedding-v4）+ 稀疏向量（BGE-M3）
  2. 稠密检索 + 稀疏检索同时执行（ANN + BM25）
  3. 结果合并去重，按 score 降序排列
  4. 支持按 content_type / region 过滤
"""
from __future__ import annotations

from loguru import logger
from pymilvus import Collection, AnnSearchRequest, RRFRanker

from app.utils.milvus_client import get_collection
from app.utils.embedder import embed_dense, embed_sparse


async def hybrid_search(
    query: str,
    top_k: int = 20,
    content_type: str | None = None,
    region: str | None = None,
) -> list[dict]:
    """
    对单条查询执行混合检索，返回最多 top_k 条原始结果。

    参数：
        query        - 查询文本
        top_k        - 返回结果数量
        content_type - 可选过滤：景点介绍 / 线路推荐 / 酒店信息 / 美食推荐 / 交通指南
        region       - 可选过滤：地区/城市名称

    返回：[{"content": str, "score": float, "metadata": dict}, ...]
    """
    # 1. 生成查询向量（稀疏同步调用避免线程安全问题，稠密异步调用 API）
    sparse_results = embed_sparse([query])
    dense_results = await embed_dense([query])

    dense_vec = dense_results[0]
    sparse_vec = sparse_results[0]

    # 2. 构建 Milvus 过滤表达式
    filters = []
    if content_type:
        filters.append(f'content_type == "{content_type}"')
    if region:
        filters.append(f'region like "%{region}%"')
    expr = " && ".join(filters) if filters else ""

    # 3. 执行混合检索
    collection = get_collection()
    output_fields = [
        "id", "content", "content_type", "attraction_name",
        "route_name", "hotel_name", "restaurant_name", "region", "source_file",
    ]

    try:
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 16}},
            limit=top_k,
            expr=expr or None,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k,
            expr=expr or None,
        )
        # RRF 融合排序
        ranker = RRFRanker(k=60)
        results = collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=ranker,
            limit=top_k,
            output_fields=output_fields,
        )
    except Exception as e:
        logger.warning(f"混合检索失败，降级为纯稠密检索: {e}")
        results = _dense_only_search(collection, dense_vec, top_k, expr, output_fields)

    # 4. 格式化输出
    hits = results[0] if results else []
    docs = []
    for hit in hits:
        entity = hit.entity
        docs.append({
            "content":  entity.get("content", ""),
            "score":    hit.score,
            "metadata": {
                "id":               entity.get("id", ""),
                "content_type":     entity.get("content_type", ""),
                "attraction_name":  entity.get("attraction_name", ""),
                "route_name":       entity.get("route_name", ""),
                "hotel_name":       entity.get("hotel_name", ""),
                "restaurant_name":  entity.get("restaurant_name", ""),
                "region":           entity.get("region", ""),
                "source_file":      entity.get("source_file", ""),
            },
        })

    logger.info(f"混合检索完成: query='{query[:30]}...' 召回 {len(docs)} 条")
    return docs


def _dense_only_search(
    collection: Collection,
    dense_vec: list[float],
    top_k: int,
    expr: str,
    output_fields: list[str],
) -> list:
    """降级方案：仅使用稠密向量检索"""
    search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
    return collection.search(
        data=[dense_vec],
        anns_field="dense_vector",
        param=search_params,
        limit=top_k,
        expr=expr or None,
        output_fields=output_fields,
    )
