"""
Web 搜索工具（第三路召回）

使用 Tavily Search API，专为 RAG 场景设计，返回干净的正文片段。
仅当 .env 中配置了 TAVILY_API_KEY 时才启用；未配置时返回空列表，
不影响已有的向量检索路径。

免费注册：https://app.tavily.com  （1000次/月免费额度）
"""
from __future__ import annotations

from loguru import logger

from app.config import settings

# 懒加载，避免未安装 tavily-python 时启动报错
_tavily_client = None


def _get_client():
    global _tavily_client
    if _tavily_client is None:
        try:
            from tavily import TavilyClient          # type: ignore
            _tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        except ImportError:
            logger.warning("tavily-python 未安装，Web 搜索不可用。运行 pip install tavily-python")
            _tavily_client = False   # 标记为不可用，避免重复尝试
    return _tavily_client if _tavily_client else None


def web_search_enabled() -> bool:
    """判断 Web 搜索是否可用（配置了 API Key 且包已安装）"""
    return bool(settings.TAVILY_API_KEY)


async def web_search(query: str, max_results: int | None = None) -> list[dict]:
    """
    执行 Web 搜索，结果格式与向量检索保持一致：
      [{"content": str, "score": float, "metadata": dict}, ...]

    未配置 TAVILY_API_KEY 时直接返回 []。
    """
    if not web_search_enabled():
        return []

    n = max_results or settings.WEB_SEARCH_MAX_RESULTS
    client = _get_client()
    if client is None:
        return []

    try:
        import asyncio
        loop = asyncio.get_running_loop()
        # Tavily 客户端目前是同步的，放线程池避免阻塞事件循环
        resp = await loop.run_in_executor(
            None,
            lambda: client.search(
                query=query,
                max_results=n,
                search_depth="advanced",   # 更深度搜索，质量更高
                include_raw_content=False,
            ),
        )
        results = resp.get("results", [])
        docs = []
        for r in results:
            content = r.get("content", "") or r.get("snippet", "")
            if not content:
                continue
            docs.append({
                "content": content,
                "score":   r.get("score", 0.5),   # Tavily 自带相关性分数 0~1
                "metadata": {
                    "id":              f"web::{r.get('url', '')}",
                    "content_type":    "web",
                    "source_file":     r.get("url", ""),
                    "attraction_name": "",
                    "route_name":      "",
                    "hotel_name":      "",
                    "restaurant_name": "",
                    "region":          "",
                    "title":           r.get("title", ""),
                },
            })
        logger.info(f"Web 搜索完成: query='{query[:30]}…' 召回 {len(docs)} 条")
        return docs

    except Exception as e:
        logger.warning(f"Web 搜索失败（不影响本地检索）: {e}")
        return []
