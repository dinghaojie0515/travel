"""
文档导入 LangGraph 工作流

节点顺序：
  load_file → chunk_text → embed_chunks → upsert_milvus → upload_minio

进度追踪：每个节点执行前后调用 update_progress()，embed 阶段细报批次。
"""
from __future__ import annotations

import io
from typing import TypedDict, Callable
from langgraph.graph import StateGraph, START, END
from loguru import logger

from app.utils.doc_parser import parse_markdown, chunk_text as do_chunk
from app.utils.embedder import embed_chunks as do_embed
from app.utils.milvus_client import get_collection
from app.utils.minio_client import get_minio_client
from app.config import settings


# ============================================================
# 进度注册表（task_id → 进度信息）
# ============================================================
_progress_registry: dict[str, dict] = {}

STEP_LABELS = {
    "load_file":     "解析文档",
    "chunk_text":    "智能分块",
    "embed_sparse":  "稀疏向量化",
    "embed_dense":   "稠密向量化",
    "upsert_milvus": "写入向量库",
    "upload_minio":  "存储原文",
}


def update_progress(task_id: str, step: str, pct: int, detail: str = "") -> None:
    """更新任务进度（由各节点调用）"""
    _progress_registry[task_id] = {
        "current_step":  step,
        "step_label":    STEP_LABELS.get(step, step),
        "progress":      min(max(pct, 0), 100),
        "step_detail":   detail,
    }
    logger.info(f"[{task_id}] 进度 {pct}% | {STEP_LABELS.get(step, step)}"
                + (f" | {detail}" if detail else ""))


def get_task_progress(task_id: str) -> dict:
    """获取任务实时进度（供 API 层查询）"""
    return _progress_registry.get(task_id, {})


def clear_progress(task_id: str) -> None:
    _progress_registry.pop(task_id, None)


# ============================================================
# State 定义
# ============================================================
class IngestState(TypedDict):
    task_id:        str
    file_name:      str
    file_bytes:     bytes
    content:        str
    metadata:       dict
    chunks:         list[dict]
    dense_vectors:  list[list[float]]
    sparse_vectors: list[dict]
    status:         str
    error:          str | None
    chunk_count:    int


# ============================================================
# 节点函数
# ============================================================
def node_load_file(state: IngestState) -> dict:
    tid = state["task_id"]
    update_progress(tid, "load_file", 5, f"正在解析 {state['file_name']}")
    content, metadata = parse_markdown(state["file_bytes"], state["file_name"])
    update_progress(tid, "load_file", 10, f"解析完成，{len(content)} 字符")
    return {"content": content, "metadata": metadata, "status": "running"}


def node_chunk_text(state: IngestState) -> dict:
    tid = state["task_id"]
    update_progress(tid, "chunk_text", 12, "正在分块...")
    chunks = do_chunk(state["content"], state["metadata"])
    update_progress(tid, "chunk_text", 20, f"分块完成，共 {len(chunks)} 个片段")
    return {"chunks": chunks}


async def node_embed_chunks(state: IngestState) -> dict:
    tid = state["task_id"]
    chunks = state["chunks"]
    total = len(chunks)
    sparse_batch = 16
    dense_batch  = 10

    # ---- 稀疏向量：BGE-M3 ----
    total_sparse_batches = (total + sparse_batch - 1) // sparse_batch
    update_progress(tid, "embed_sparse", 20, f"稀疏向量化，共 {total_sparse_batches} 批")

    from app.utils.embedder import embed_sparse, embed_dense

    def sparse_progress(batch_idx: int, total_batches: int) -> None:
        pct = 20 + int((batch_idx / total_batches) * 30)
        update_progress(tid, "embed_sparse",
                        pct, f"稀疏向量化 {batch_idx}/{total_batches} 批")

    texts = [c["content"] for c in chunks]
    sparse_vecs = embed_sparse(texts, batch_size=sparse_batch,
                               on_progress=sparse_progress)

    # ---- 稠密向量：API ----
    total_dense_batches = (total + dense_batch - 1) // dense_batch
    update_progress(tid, "embed_dense", 50, f"稠密向量化，共 {total_dense_batches} 批")

    async def dense_progress(batch_idx: int, total_batches: int) -> None:
        pct = 50 + int((batch_idx / total_batches) * 30)
        update_progress(tid, "embed_dense",
                        pct, f"稠密向量化 {batch_idx}/{total_batches} 批")

    dense_vecs = await embed_dense(texts, batch_size=dense_batch,
                                   on_progress=dense_progress)

    update_progress(tid, "embed_dense", 80, f"向量化完成，{total} 条")
    return {"dense_vectors": dense_vecs, "sparse_vectors": sparse_vecs}


def node_upsert_milvus(state: IngestState) -> dict:
    tid = state["task_id"]
    chunks     = state["chunks"]
    dense_vecs = state["dense_vectors"]
    sparse_vecs= state["sparse_vectors"]

    update_progress(tid, "upsert_milvus", 82, f"正在写入 {len(chunks)} 条到 Milvus...")
    collection = get_collection()
    data = [
        {
            "id":              chunks[i]["id"],
            "content":         chunks[i]["content"][:4096],
            "content_type":    chunks[i]["content_type"],
            "attraction_name": chunks[i]["attraction_name"],
            "route_name":      chunks[i]["route_name"],
            "hotel_name":      chunks[i]["hotel_name"],
            "restaurant_name": chunks[i]["restaurant_name"],
            "region":          chunks[i]["region"],
            "source_file":     chunks[i]["source_file"],
            "dense_vector":    dense_vecs[i],
            "sparse_vector":   sparse_vecs[i],
        }
        for i in range(len(chunks))
    ]
    collection.upsert(data)
    update_progress(tid, "upsert_milvus", 90, f"Milvus 写入完成，{len(chunks)} 条")
    return {"chunk_count": len(chunks)}


def node_upload_minio(state: IngestState) -> dict:
    tid = state["task_id"]
    update_progress(tid, "upload_minio", 92, "正在上传原文到 MinIO...")
    client = get_minio_client()
    file_data   = io.BytesIO(state["file_bytes"])
    object_name = f"docs/{tid}/{state['file_name']}"
    client.put_object(
        bucket_name  = settings.MINIO_BUCKET,
        object_name  = object_name,
        data         = file_data,
        length       = len(state["file_bytes"]),
        content_type = "text/markdown",
    )
    update_progress(tid, "upload_minio", 100, "导入完成")
    return {"status": "success"}


def node_handle_error(state: IngestState) -> dict:
    tid = state["task_id"]
    err = state.get("error", "未知错误")
    logger.error(f"[{tid}] 导入失败: {err}")
    _progress_registry.setdefault(tid, {})["step_detail"] = f"失败: {err}"
    return {"status": "failed"}


# ============================================================
# _safe 包装 & 路由
# ============================================================
def _safe(node_fn):
    import functools, asyncio

    if asyncio.iscoroutinefunction(node_fn):
        @functools.wraps(node_fn)
        async def async_wrapper(state: IngestState) -> dict:
            try:
                return await node_fn(state)
            except Exception as e:
                logger.exception(f"节点 {node_fn.__name__} 异常")
                return {"status": "failed", "error": str(e)}
        return async_wrapper
    else:
        @functools.wraps(node_fn)
        def sync_wrapper(state: IngestState) -> dict:
            try:
                return node_fn(state)
            except Exception as e:
                logger.exception(f"节点 {node_fn.__name__} 异常")
                return {"status": "failed", "error": str(e)}
        return sync_wrapper


def _route_on_error(state: IngestState) -> str:
    return "handle_error" if state.get("status") == "failed" else "continue"


# ============================================================
# 构建 Graph
# ============================================================
def build_ingest_graph():
    graph = StateGraph(IngestState)

    graph.add_node("load_file",     _safe(node_load_file))
    graph.add_node("chunk_text",    _safe(node_chunk_text))
    graph.add_node("embed_chunks",  _safe(node_embed_chunks))
    graph.add_node("upsert_milvus", _safe(node_upsert_milvus))
    graph.add_node("upload_minio",  _safe(node_upload_minio))
    graph.add_node("handle_error",  node_handle_error)

    graph.add_edge(START, "load_file")

    for src, dst in [
        ("load_file",     "chunk_text"),
        ("chunk_text",    "embed_chunks"),
        ("embed_chunks",  "upsert_milvus"),
        ("upsert_milvus", "upload_minio"),
        ("upload_minio",  END),
    ]:
        if dst is END:
            graph.add_conditional_edges(
                src, _route_on_error,
                {"continue": END, "handle_error": "handle_error"},
            )
        else:
            graph.add_conditional_edges(
                src, _route_on_error,
                {"continue": dst, "handle_error": "handle_error"},
            )

    graph.add_edge("handle_error", END)
    return graph.compile()


ingest_graph = build_ingest_graph()
