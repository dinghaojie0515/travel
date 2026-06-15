from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional
from loguru import logger

from app.processor.retrieval_graph import retrieve

router = APIRouter()


class SearchResult(BaseModel):
    content: str
    score: float
    rerank_score: float = 0.0
    metadata: dict


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int
    query: str


@router.get("/", response_model=SearchResponse, summary="知识内容检索（混合检索 + 重排序）")
async def search(
    q: str = Query(..., description="搜索关键词或问题"),
    content_type: Optional[str] = Query(None, description="内容类型过滤：景点介绍/线路推荐/酒店信息/美食推荐/交通指南"),
    region: Optional[str] = Query(None, description="地区/城市过滤"),
    top_k: int = Query(6, description="返回结果数量"),
):
    """
    混合检索旅游知识内容（稠密 + 稀疏 BM25 + HyDE + 重排序），支持按内容类型和地区过滤
    """
    logger.info(f"收到检索请求: query={q}, type={content_type}, region={region}")
    docs = await retrieve(query=q, top_k=top_k, content_type=content_type, region=region)
    results = [
        SearchResult(
            content=doc["content"],
            score=doc.get("score", 0.0),
            rerank_score=doc.get("rerank_score", 0.0),
            metadata=doc.get("metadata", {}),
        )
        for doc in docs
    ]
    return SearchResponse(results=results, total=len(results), query=q)
