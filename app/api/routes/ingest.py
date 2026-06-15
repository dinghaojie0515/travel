"""
文档导入 API

POST /api/ingest/upload  上传 Markdown 文件，异步执行导入流水线
GET  /api/ingest/status/{task_id}  查询任务状态
GET  /api/ingest/tasks   查询所有任务列表
"""
import asyncio
import uuid
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from loguru import logger

from app.processor.ingest_graph import ingest_graph, get_task_progress, clear_progress

router = APIRouter()

_task_store: dict[str, dict] = {}


class IngestResponse(BaseModel):
    task_id:   str
    status:    str
    message:   str
    file_name: str


class TaskStatus(BaseModel):
    task_id:      str
    file_name:    str
    status:       str           # pending / running / success / failed
    progress:     int = 0       # 0 ~ 100
    current_step: str = ""      # 当前节点 key
    step_label:   str = ""      # 当前节点中文名
    step_detail:  str = ""      # 批次等细节说明
    chunk_count:  int = 0
    error:        str | None = None
    created_at:   str
    updated_at:   str


# ----------------------------------------------------------------
# 后台任务：执行 LangGraph 导入流水线
# ----------------------------------------------------------------
async def _run_ingest(task_id: str, file_name: str, file_bytes: bytes):
    _task_store[task_id]["status"] = "running"
    _task_store[task_id]["updated_at"] = _now()

    initial_state = {
        "task_id":       task_id,
        "file_name":     file_name,
        "file_bytes":    file_bytes,
        "content":       "",
        "metadata":      {},
        "chunks":        [],
        "dense_vectors": [],
        "sparse_vectors":[],
        "status":        "running",
        "error":         None,
        "chunk_count":   0,
    }

    try:
        result = await ingest_graph.ainvoke(initial_state)
        final_status = result.get("status", "success")
        _task_store[task_id].update({
            "status":      final_status,
            "progress":    100 if final_status == "success" else _task_store[task_id].get("progress", 0),
            "step_label":  "导入完成" if final_status == "success" else "导入失败",
            "step_detail": f"共 {result.get('chunk_count', 0)} 个知识块" if final_status == "success" else result.get("error", ""),
            "chunk_count": result.get("chunk_count", 0),
            "error":       result.get("error"),
            "updated_at":  _now(),
        })
        logger.info(f"任务 {task_id} 完成: status={final_status}, chunks={result.get('chunk_count')}")
    except Exception as e:
        logger.exception(f"任务 {task_id} 异常")
        _task_store[task_id].update({
            "status":     "failed",
            "step_label": "导入失败",
            "step_detail": str(e),
            "error":      str(e),
            "updated_at": _now(),
        })
    finally:
        clear_progress(task_id)


# ----------------------------------------------------------------
# 路由
# ----------------------------------------------------------------
@router.post("/upload", response_model=IngestResponse, summary="上传并导入知识文档")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    上传 Markdown 格式旅游知识文档，系统自动完成：
    解析 → 分块 → 向量化（稠密+稀疏）→ 写入 Milvus → 原文存入 MinIO

    **元数据** 可在文件开头用 YAML frontmatter 声明，例如：
    ```yaml
    ---
    content_type: 景点介绍
    attraction_name: 天涯海角
    region: 三亚
    ---
    ```
    未声明时默认为"景点介绍"，其余字段置空。
    """
    if not file.filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="仅支持 Markdown (.md) 格式文件")

    task_id = str(uuid.uuid4())
    file_bytes = await file.read()

    _task_store[task_id] = {
        "task_id":      task_id,
        "file_name":    file.filename,
        "status":       "pending",
        "progress":     0,
        "current_step": "",
        "step_label":   "等待处理",
        "step_detail":  "",
        "chunk_count":  0,
        "error":        None,
        "created_at":   _now(),
        "updated_at":   _now(),
    }

    background_tasks.add_task(_run_ingest, task_id, file.filename, file_bytes)
    logger.info(f"任务 {task_id} 已创建: {file.filename}")

    return IngestResponse(
        task_id=task_id,
        status="pending",
        message="文档已接收，正在后台处理，请通过 /status/{task_id} 查询进度",
        file_name=file.filename,
    )


@router.get("/status/{task_id}", response_model=TaskStatus, summary="查询导入任务状态（含进度）")
async def get_task_status(task_id: str):
    """查询文档导入任务的当前状态，running 时包含实时进度"""
    task = _task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

    # 合并实时进度（仅 running 阶段有值）
    merged = dict(task)
    if task.get("status") == "running":
        prog = get_task_progress(task_id)
        merged.update({
            "progress":     prog.get("progress",     merged.get("progress", 0)),
            "current_step": prog.get("current_step", ""),
            "step_label":   prog.get("step_label",   ""),
            "step_detail":  prog.get("step_detail",  ""),
        })
    return TaskStatus(**merged)


@router.get("/tasks", summary="查询所有任务列表")
async def list_tasks():
    """返回所有导入任务的状态列表（最近100条）"""
    tasks = sorted(_task_store.values(), key=lambda t: t["created_at"], reverse=True)
    return {"total": len(tasks), "tasks": tasks[:100]}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
