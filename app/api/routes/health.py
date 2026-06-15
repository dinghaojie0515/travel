"""
健康检查接口

GET /health          简单存活检查
GET /health/services 各依赖服务连接状态
"""
from fastapi import APIRouter
from loguru import logger

router = APIRouter()


@router.get("/", summary="存活检查")
async def ping():
    return {"status": "ok", "message": "旅游知识库服务运行中"}


@router.get("/services", summary="各依赖服务连接状态")
async def check_services():
    results = {}

    # Milvus
    try:
        from pymilvus import connections
        conn = connections.get_connection_addr("default")
        results["milvus"] = {"status": "ok", "host": conn.get("host", "?")}
    except Exception as e:
        results["milvus"] = {"status": "error", "detail": str(e)}

    # MongoDB
    try:
        from app.utils.mongo_client import get_mongo_db
        db = get_mongo_db()
        db.command("ping")
        results["mongodb"] = {"status": "ok"}
    except Exception as e:
        results["mongodb"] = {"status": "error", "detail": str(e)}

    # MinIO
    try:
        from app.utils.minio_client import get_minio_client
        from app.config import settings
        client = get_minio_client()
        client.bucket_exists(settings.MINIO_BUCKET)
        results["minio"] = {"status": "ok", "endpoint": settings.MINIO_ENDPOINT}
    except Exception as e:
        results["minio"] = {"status": "error", "detail": str(e)}

    overall = "ok" if all(v["status"] == "ok" for v in results.values()) else "degraded"
    return {"status": overall, "services": results}
