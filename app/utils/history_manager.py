"""
对话历史管理（MongoDB）

每条历史记录格式：
{
    "session_id": str,
    "role":       "user" | "assistant",
    "content":    str,
    "created_at": datetime,
    "sources":    list[dict]   # assistant 消息携带的引用来源
}
"""
from __future__ import annotations

from datetime import datetime, timezone
from loguru import logger
from app.utils.mongo_client import get_history_collection


def save_message(
    session_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> None:
    """将单条消息写入 MongoDB"""
    col = get_history_collection()
    doc = {
        "session_id": session_id,
        "role":       role,
        "content":    content,
        "created_at": datetime.now(timezone.utc),
        "sources":    sources or [],
    }
    col.insert_one(doc)


def load_history(session_id: str, max_turns: int = 10) -> list[dict]:
    """
    加载会话历史，返回最近 max_turns 轮（每轮含 user + assistant 两条）。
    返回格式与 OpenAI messages 兼容：[{"role": "user"|"assistant", "content": str}, ...]
    """
    col = get_history_collection()
    docs = list(
        col.find(
            {"session_id": session_id},
            {"_id": 0, "role": 1, "content": 1},
        )
        .sort("created_at", 1)      # 按时间升序
        .limit(max_turns * 2)       # 每轮 2 条，取最近 N 轮
    )
    # 只保留 role / content，去掉 MongoDB 其他字段
    return [{"role": d["role"], "content": d["content"]} for d in docs]


def delete_session(session_id: str) -> int:
    """删除整个会话的历史记录，返回删除条数"""
    col = get_history_collection()
    result = col.delete_many({"session_id": session_id})
    logger.info(f"删除会话 {session_id} 历史，共 {result.deleted_count} 条")
    return result.deleted_count


def list_sessions(limit: int = 50) -> list[dict]:
    """列出最近 N 个有历史的 session_id 及最后一条消息时间"""
    col = get_history_collection()
    pipeline = [
        {"$sort": {"created_at": -1}},
        {"$group": {
            "_id": "$session_id",
            "last_active": {"$first": "$created_at"},
            "message_count": {"$sum": 1},
        }},
        {"$sort": {"last_active": -1}},
        {"$limit": limit},
        {"$project": {
            "_id": 0,
            "session_id": "$_id",
            "last_active": 1,
            "message_count": 1,
        }},
    ]
    return list(col.aggregate(pipeline))
