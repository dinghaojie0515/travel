from typing import Optional
from minio import Minio
from minio.error import S3Error
from loguru import logger
from app.config import settings

_client: Optional[Minio] = None


async def init_minio():
    """初始化 MinIO 连接并确保 Bucket 存在"""
    global _client
    try:
        _client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        bucket = settings.MINIO_BUCKET
        if not _client.bucket_exists(bucket):
            _client.make_bucket(bucket)
            logger.info(f"MinIO Bucket '{bucket}' 创建完成")
        else:
            logger.info(f"MinIO Bucket '{bucket}' 已存在")
        logger.info(f"MinIO 连接成功: {settings.MINIO_ENDPOINT}")
    except S3Error as e:
        logger.error(f"MinIO 初始化失败: {e}")
        raise


def get_minio_client() -> Minio:
    """获取 MinIO 客户端实例"""
    if _client is None:
        raise RuntimeError("MinIO 客户端未初始化，请确认 init_minio() 已调用")
    return _client
