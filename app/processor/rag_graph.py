"""
RAG 问答 LangGraph 工作流

节点顺序：
  load_history → retrieve_docs → build_prompt → generate → save_history

流式输出由 API 层直接调用 LLM，本 Graph 负责非流式的完整答案生成；
流式版本通过 rag_stream() 函数对外暴露，逐阶段 yield step 事件供前端追踪。
"""
from __future__ import annotations

import asyncio
import uuid
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from loguru import logger

from app.utils.history_manager import load_history, save_message
from app.processor.retrieval_graph import retrieve
from app.utils.llm_client import chat_completion, generate_hyde_doc
from app.utils.retriever import hybrid_search
from app.utils.reranker import rerank
from app.utils.web_searcher import web_search, web_search_enabled


# ============================================================
# State 定义
# ============================================================
class RAGState(TypedDict):
    session_id:   str
    question:     str
    history:      list[dict]   # [{role, content}, ...]
    docs:         list[dict]   # reranked_docs
    prompt_msgs:  list[dict]   # 完整 messages 列表
    answer:       str
    sources:      list[dict]   # 引用来源


# ============================================================
# 系统提示
# ============================================================
SYSTEM_PROMPT = """【重要语言规则】你的所有回答必须使用简体中文，严禁出现任何繁体字。参考资料中如有繁体字，须自动转换为简体中文输出。

你是一位专业的旅游顾问助手，基于提供的旅游知识库内容回答用户问题。

回答要求：
- 全程使用简体中文，不得出现繁体字（如"景點"应写成"景点"，"連絡"应写成"联络"）
- 优先使用知识库中的内容作答，内容要准确、具体
- 如果知识库中没有相关信息，请如实告知，不要编造
- 回答语气亲切自然，适合旅游咨询场景
- 如有多个景点或建议，请用列表形式展示，条理清晰

知识库参考内容：
{context}
"""


def _build_context(docs: list[dict]) -> str:
    """将检索到的文档块拼接成上下文字符串"""
    if not docs:
        return "（暂无相关知识库内容）"
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.get("metadata", {})
        source_info = []
        if meta.get("attraction_name"):
            source_info.append(f"景点：{meta['attraction_name']}")
        if meta.get("region"):
            source_info.append(f"地区：{meta['region']}")
        if meta.get("source_file"):
            source_info.append(f"来源：{meta['source_file']}")
        header = f"[{i}] " + " | ".join(source_info) if source_info else f"[{i}]"
        parts.append(f"{header}\n{doc['content']}")
    return "\n\n".join(parts)


def _extract_sources(docs: list[dict]) -> list[dict]:
    """从 reranked_docs 提取引用来源信息（含内容摘要）"""
    sources = []
    seen_ids = set()
    for doc in docs:
        meta   = doc.get("metadata", {})
        doc_id = meta.get("id", "")
        if doc_id and doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        # 取内容前 120 字作为摘要展示
        snippet = doc.get("content", "")[:120].strip()
        if len(doc.get("content", "")) > 120:
            snippet += "…"
        sources.append({
            "content_type":    meta.get("content_type", ""),
            "attraction_name": meta.get("attraction_name", ""),
            "region":          meta.get("region", ""),
            "source_file":     meta.get("source_file", ""),
            "rerank_score":    round(doc.get("rerank_score", 0.0), 4),
            "snippet":         snippet,
        })
    return sources


# ============================================================
# 节点函数
# ============================================================
def node_load_history(state: RAGState) -> dict:
    """节点1：从 MongoDB 加载会话历史"""
    history = load_history(state["session_id"], max_turns=10)
    logger.info(f"[{state['session_id']}] 加载历史 {len(history)} 条")
    return {"history": history}


async def node_retrieve_docs(state: RAGState) -> dict:
    """节点2：检索相关知识块"""
    docs = await retrieve(query=state["question"], top_k=6)
    logger.info(f"[{state['session_id']}] 检索到 {len(docs)} 条相关文档")
    return {"docs": docs, "sources": _extract_sources(docs)}


def node_build_prompt(state: RAGState) -> dict:
    """节点3：构造完整 messages（系统提示 + 历史 + 当前问题）"""
    context = _build_context(state["docs"])
    system_msg = {"role": "system", "content": SYSTEM_PROMPT.format(context=context)}

    # 历史消息（最近 10 轮）
    history = state.get("history", [])

    # 当前问题
    user_msg = {"role": "user", "content": state["question"]}

    prompt_msgs = [system_msg] + history + [user_msg]
    logger.debug(f"[{state['session_id']}] Prompt 共 {len(prompt_msgs)} 条消息")
    return {"prompt_msgs": prompt_msgs}


async def node_generate(state: RAGState) -> dict:
    """节点4：调用 Qwen 生成完整答案（非流式，供内部使用）"""
    answer = await chat_completion(
        messages=state["prompt_msgs"],
        temperature=0.7,
        max_tokens=2048,
        stream=False,
    )
    logger.info(f"[{state['session_id']}] 生成完成，答案长度={len(answer)}")
    return {"answer": answer}


def node_save_history(state: RAGState) -> dict:
    """节点5：将本轮问答存入 MongoDB"""
    save_message(state["session_id"], "user", state["question"])
    save_message(state["session_id"], "assistant", state["answer"], sources=state["sources"])
    logger.info(f"[{state['session_id']}] 历史已保存")
    return {}


# ============================================================
# 构建 Graph
# ============================================================
def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("load_history",   node_load_history)
    graph.add_node("retrieve_docs",  node_retrieve_docs)
    graph.add_node("build_prompt",   node_build_prompt)
    graph.add_node("generate",       node_generate)
    graph.add_node("save_history",   node_save_history)

    graph.add_edge(START,            "load_history")
    graph.add_edge("load_history",   "retrieve_docs")
    graph.add_edge("retrieve_docs",  "build_prompt")
    graph.add_edge("build_prompt",   "generate")
    graph.add_edge("generate",       "save_history")
    graph.add_edge("save_history",   END)

    return graph.compile()


rag_graph = build_rag_graph()


# ============================================================
# 流式生成入口（供 API 层 SSE 使用）
# ============================================================
async def rag_stream(session_id: str, question: str):
    """
    流式问答入口，按阶段 yield 以下事件类型：
      step    — 处理阶段状态（running / done / error）
      sources — 引用来源列表
      text    — LLM 生成的文字片段
    """

    # ── 阶段 1: 加载对话历史 ──────────────────────────────────
    yield {"type": "step", "id": "history", "icon": "📚",
           "label": "加载对话历史", "status": "running"}
    history = load_history(session_id, max_turns=10)
    yield {"type": "step", "id": "history", "icon": "📚",
           "label": "加载对话历史", "status": "done",
           "detail": f"{len(history) // 2} 轮"}

    # ── 阶段 2: 问题理解与 HyDE 扩写 ─────────────────────────
    yield {"type": "step", "id": "hyde", "icon": "✂️",
           "label": "问题切片与扩写", "status": "running"}
    try:
        hyde_doc = await generate_hyde_doc(question)
    except Exception:
        hyde_doc = question
    yield {"type": "step", "id": "hyde", "icon": "✂️",
           "label": "问题切片与扩写", "status": "done"}

    # ── 阶段 3: 并发检索（知识库 + 可选 Web）────────────────
    yield {"type": "step", "id": "vector", "icon": "🔍",
           "label": "知识库检索", "status": "running"}
    has_web = web_search_enabled()
    if has_web:
        yield {"type": "step", "id": "web", "icon": "🌐",
               "label": "Web 实时搜索", "status": "running"}

    tasks = [
        hybrid_search(question, top_k=20),
        hybrid_search(hyde_doc,  top_k=20),
    ]
    if has_web:
        tasks.append(web_search(question))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    original_docs = results[0] if not isinstance(results[0], Exception) else []
    hyde_docs     = results[1] if not isinstance(results[1], Exception) else []
    web_docs      = (results[2]
                     if len(results) > 2 and not isinstance(results[2], Exception)
                     else [])

    if isinstance(results[0], Exception):
        logger.warning(f"原始向量检索异常: {results[0]}")
    if isinstance(results[1], Exception):
        logger.warning(f"HyDE 向量检索异常: {results[1]}")
    if len(results) > 2 and isinstance(results[2], Exception):
        logger.warning(f"Web 搜索异常: {results[2]}")

    vec_count = len(original_docs) + len(hyde_docs)
    yield {"type": "step", "id": "vector", "icon": "🔍",
           "label": "知识库检索", "status": "done",
           "detail": f"召回 {vec_count} 条"}
    if has_web:
        yield {"type": "step", "id": "web", "icon": "🌐",
               "label": "Web 实时搜索", "status": "done",
               "detail": f"获取 {len(web_docs)} 条"}

    # ── 阶段 4: 结果融合与重排 ────────────────────────────────
    yield {"type": "step", "id": "rerank", "icon": "🔀",
           "label": "结果融合重排", "status": "running"}

    all_docs = original_docs + hyde_docs + web_docs
    seen: dict = {}
    for doc in all_docs:
        doc_id = doc["metadata"].get("id") or doc["content"][:48]
        if doc_id not in seen or doc["score"] > seen[doc_id]["score"]:
            seen[doc_id] = doc
    merged = sorted(seen.values(), key=lambda d: d["score"], reverse=True)

    docs    = rerank(query=question, docs=merged, top_k=6)
    sources = _extract_sources(docs)

    yield {"type": "step", "id": "rerank", "icon": "🔀",
           "label": "结果融合重排", "status": "done",
           "detail": f"精选 {len(docs)} 条"}

    # ── 阶段 5: 构造 Prompt 并流式生成 ───────────────────────
    context     = _build_context(docs)
    system_msg  = {"role": "system", "content": SYSTEM_PROMPT.format(context=context)}
    user_msg    = {"role": "user",   "content": question}
    prompt_msgs = [system_msg] + history + [user_msg]

    yield {"type": "sources", "sources": sources}
    yield {"type": "step", "id": "generate", "icon": "💡",
           "label": "生成回答", "status": "running"}

    full_answer: list[str] = []
    stream = await chat_completion(
        messages=prompt_msgs,
        temperature=0.7,
        max_tokens=2048,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            full_answer.append(delta)
            yield {"type": "text", "content": delta}

    yield {"type": "step", "id": "generate", "icon": "💡",
           "label": "生成回答", "status": "done"}

    # ── 保存历史 ──────────────────────────────────────────────
    answer_text = "".join(full_answer)
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer_text, sources=sources)
    logger.info(f"[{session_id}] 流式回答完成，长度={len(answer_text)}")
