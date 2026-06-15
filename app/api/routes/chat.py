"""
知识问答 API

POST /api/chat/stream       流式问答（SSE，推荐）
POST /api/chat/             非流式问答（一次性返回完整答案）
GET  /api/chat/history/{session_id}   获取会话历史
DELETE /api/chat/history/{session_id} 清空会话历史
GET  /api/chat/sessions    获取所有会话列表
"""
import uuid
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from loguru import logger

from app.processor.rag_graph import rag_graph, rag_stream, RAGState
from app.utils.history_manager import load_history, delete_session, list_sessions

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list = []


# ----------------------------------------------------------------
# 流式问答（SSE）
# ----------------------------------------------------------------
@router.post("/stream", summary="流式问答（SSE 实时推送）")
async def chat_stream(request: ChatRequest):
    """
    基于知识库的流式问答，通过 SSE 协议逐字推送答案。

    前端接收方式：
    ```javascript
    const es = new EventSource('/api/chat/stream');
    es.onmessage = e => console.log(e.data);
    ```

    每个 SSE 事件格式：
    - `data: <文字片段>\\n\\n`  — 正常答案片段
    - `data: [DONE]\\n\\n`      — 答案生成完毕
    - `data: [ERROR] <msg>\\n\\n` — 发生错误
    """
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"流式问答请求: session={session_id}, q={request.question[:40]}")

    async def event_generator():
        # 先推送 session_id，方便前端保存
        yield f"data: {json.dumps({'type': 'session_id', 'session_id': session_id}, ensure_ascii=False)}\n\n"
        try:
            # rag_stream 现在 yield dict，包含 type: sources / type: text
            async for event in rag_stream(session_id, request.question):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception(f"流式生成异常: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ----------------------------------------------------------------
# 非流式问答
# ----------------------------------------------------------------
@router.post("/", response_model=ChatResponse, summary="非流式问答（一次返回完整答案）")
async def chat(request: ChatRequest):
    """
    基于知识库内容回答用户问题，支持多轮对话（通过 session_id 关联历史）。
    """
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"非流式问答请求: session={session_id}, q={request.question[:40]}")

    initial_state: RAGState = {
        "session_id":  session_id,
        "question":    request.question,
        "history":     [],
        "docs":        [],
        "prompt_msgs": [],
        "answer":      "",
        "sources":     [],
    }

    result = await rag_graph.ainvoke(initial_state)

    return ChatResponse(
        answer=result["answer"],
        session_id=session_id,
        sources=result.get("sources", []),
    )


# ----------------------------------------------------------------
# 会话历史管理
# ----------------------------------------------------------------
@router.get("/history/{session_id}", summary="获取会话历史")
async def get_history(session_id: str, max_turns: int = 20):
    """获取指定会话的对话历史记录"""
    messages = load_history(session_id, max_turns=max_turns)
    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": messages,
    }


@router.delete("/history/{session_id}", summary="清空会话历史")
async def clear_history(session_id: str):
    """清空指定会话的所有历史记录"""
    count = delete_session(session_id)
    return {"session_id": session_id, "deleted_count": count, "message": "会话历史已清空"}


@router.get("/sessions", summary="获取所有会话列表")
async def get_sessions(limit: int = 50):
    """获取最近有活动的会话列表"""
    sessions = list_sessions(limit=limit)
    return {"total": len(sessions), "sessions": sessions}
