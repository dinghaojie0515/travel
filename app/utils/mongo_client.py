from typing import Optional
from pymongo import MongoClient
from pymongo.collection import Collection as MongoCollection
from loguru import logger
from app.config import settings

_client: Optional[MongoClient] = None


def get_mongo_db():
    """获取 MongoDB 数据库实例（懒初始化）"""
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGODB_URI)
        logger.info("MongoDB 连接成功")
    return _client[settings.MONGODB_DB]


def get_history_collection() -> MongoCollection:
    """获取对话历史 Collection"""
    db = get_mongo_db()
    return db[settings.MONGODB_COLLECTION_HISTORY]
