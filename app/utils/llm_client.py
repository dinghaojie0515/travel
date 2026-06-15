"""
LLM 工具封装（阿里云百炼 Qwen，兼容 OpenAI 格式）
"""
from __future__ import annotations

from openai import AsyncOpenAI
from loguru import logger

from app.config import settings

_client: AsyncOpenAI | None = None


def get_llm_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_BASE_URL,
        )
    return _client


async def chat_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    stream: bool = False,
):
    """
    调用 Qwen 生成回复。

    stream=False 时返回完整字符串；
    stream=True  时返回 AsyncStream 对象（由调用方 async for 消费）。
    """
    client = get_llm_client()
    model = model or settings.LLM_MODEL

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )

    if stream:
        return response

    content = response.choices[0].message.content
    return content


async def generate_hyde_doc(query: str) -> str:
    """
    HyDE：根据用户问题生成一段「假设性答案文档」，
    用该文档的向量代替问题向量去检索，提升召回率。
    """
    messages = [
        {
            "role": "system",
            "content": (
                "你是一位旅游知识专家。请根据用户的问题，"
                "生成一段简短的、像真实旅游攻略一样的假设性回答（100~200字），"
                "不需要说明这是假设，直接给出内容即可。"
            ),
        },
        {"role": "user", "content": query},
    ]
    hyde_doc = await chat_completion(messages, temperature=0.5, max_tokens=256)
    logger.debug(f"HyDE 生成: {hyde_doc[:60]}...")
    return hyde_doc
