"""
向量化工具

- 稠密向量：调用阿里云百炼 text-embedding-v4（1024维，兼容 OpenAI 格式）
- 稀疏向量：本地 BGE-M3 模型生成词汇权重（BM25 风格关键词匹配）

on_progress 回调签名：(batch_idx: int, total_batches: int) -> None | Coroutine
"""
from __future__ import annotations

from typing import Optional, Callable, Awaitable
from openai import AsyncOpenAI
from FlagEmbedding import BGEM3FlagModel
from loguru import logger

from app.config import settings

ProgressCB    = Callable[[int, int], None]
AsyncProgressCB = Callable[[int, int], Awaitable[None]]

# ---------- BGE-M3 ----------
_bge_model: Optional[BGEM3FlagModel] = None


def get_bge_model() -> BGEM3FlagModel:
    global _bge_model
    if _bge_model is None:
        logger.info(f"加载 BGE-M3 模型: {settings.BGE_M3_MODEL}（首次加载较慢）...")
        _bge_model = BGEM3FlagModel(settings.BGE_M3_MODEL, use_fp16=False)
        logger.info("BGE-M3 模型加载完成")
    return _bge_model


# ---------- text-embedding-v4 ----------
_openai_client: Optional[AsyncOpenAI] = None


def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_BASE_URL,
        )
    return _openai_client


async def embed_dense(
    texts: list[str],
    batch_size: int = 10,
    on_progress: AsyncProgressCB | None = None,
) -> list[list[float]]:
    """
    调用 text-embedding-v4 生成稠密向量（1024维）。
    每批最多 10 条（API 限制）。on_progress(batch_idx, total_batches) 每批回调一次。
    """
    client = get_openai_client()
    all_vectors: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for idx, i in enumerate(range(0, len(texts), batch_size), 1):
        batch = texts[i: i + batch_size]
        logger.debug(f"稠密向量批次 {idx}/{total_batches}，共 {len(batch)} 条")
        resp = await client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=batch,
        )
        all_vectors.extend(item.embedding for item in resp.data)
        if on_progress:
            await on_progress(idx, total_batches)

    logger.info(f"稠密向量生成完成: {len(all_vectors)} 条，维度={len(all_vectors[0])}")
    return all_vectors


def embed_sparse(
    texts: list[str],
    batch_size: int = 16,
    on_progress: ProgressCB | None = None,
) -> list[dict[int, float]]:
    """
    使用 BGE-M3 生成稀疏向量。on_progress(batch_idx, total_batches) 每批回调一次。
    """
    model = get_bge_model()
    all_sparse: list[dict[int, float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for idx, i in enumerate(range(0, len(texts), batch_size), 1):
        batch = texts[i: i + batch_size]
        logger.debug(f"稀疏向量批次 {idx}/{total_batches}，共 {len(batch)} 条")
        output = model.encode(
            batch,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        for lw in output["lexical_weights"]:
            sparse_vec = {int(k): float(v) for k, v in lw.items() if float(v) > 1e-4}
            all_sparse.append(sparse_vec)
        if on_progress:
            on_progress(idx, total_batches)

    logger.info(f"稀疏向量生成完成: {len(all_sparse)} 条")
    return all_sparse


async def embed_chunks(
    chunks: list[dict],
    on_sparse_progress: ProgressCB | None = None,
    on_dense_progress:  AsyncProgressCB | None = None,
) -> tuple[list[list[float]], list[dict[int, float]]]:
    """
    对 chunk 列表生成稠密和稀疏向量（先稀疏后稠密）。
    PyTorch 模型非线程安全，不使用 run_in_executor。
    """
    texts = [c["content"] for c in chunks]
    sparse_vecs = embed_sparse(texts, on_progress=on_sparse_progress)
    dense_vecs  = await embed_dense(texts, on_progress=on_dense_progress)
    return dense_vecs, sparse_vecs
