from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from app.api.routes import ingest, chat, search, health
from app.utils.milvus_client import init_milvus
from app.utils.minio_client import init_minio
from app.config import settings


def _print_banner(host: str, port: int):
    base = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}"
    logger.info("=" * 52)
    logger.info(f"  旅游知识库 v0.1.0 已启动")
    logger.info(f"  API 文档  : {base}/docs")
    logger.info(f"  健康检查  : {base}/health/services")
    logger.info(f"  文档上传  : POST  {base}/api/ingest/upload")
    logger.info(f"  任务状态  : GET   {base}/api/ingest/tasks")
    logger.info(f"  知识检索  : GET   {base}/api/search/")
    logger.info(f"  流式问答  : POST  {base}/api/chat/stream")
    logger.info(f"  会话历史  : GET   {base}/api/chat/history/{{session_id}}")
    logger.info("=" * 52)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("正在初始化服务...")
    await init_milvus()
    await init_minio()
    _print_banner(settings.APP_HOST, settings.APP_PORT)
    yield
    logger.info("应用关闭")


app = FastAPI(
    title="旅游知识库 API",
    description="基于 RAG 技术的智能旅游知识库系统",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,  prefix="/health",     tags=["健康检查"])
app.include_router(ingest.router,  prefix="/api/ingest", tags=["文档导入"])
app.include_router(chat.router,    prefix="/api/chat",   tags=["知识问答"])
app.include_router(search.router,  prefix="/api/search", tags=["内容检索"])

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_RELOAD,
    )
