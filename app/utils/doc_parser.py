"""
Markdown 文档解析与分块工具

支持从 YAML frontmatter 中自动提取元数据（景点名、地区、内容类型等），
并使用 LangChain RecursiveCharacterTextSplitter 做语义感知分块。
"""
import uuid
import frontmatter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger


# 知识库支持的内容类型
VALID_CONTENT_TYPES = {
    "景点介绍", "线路推荐", "酒店信息", "美食推荐", "交通指南", "文化民俗"
}

# Markdown 标题分隔符，优先在标题处切分
MD_SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "]


def parse_markdown(file_bytes: bytes, file_name: str) -> tuple[str, dict]:
    """
    解析 Markdown 文件，提取正文和元数据。

    支持两种方式提供元数据：
    1. 文件开头的 YAML frontmatter（推荐）
    2. 纯文本（元数据全部置空，导入后可手动补充）

    返回: (正文文本, 元数据字典)
    """
    try:
        post = frontmatter.loads(file_bytes.decode("utf-8"))
        content = post.content.strip()
        meta = dict(post.metadata)
    except Exception:
        content = file_bytes.decode("utf-8", errors="ignore").strip()
        meta = {}

    metadata = {
        "content_type":      meta.get("content_type", "景点介绍"),
        "attraction_name":   meta.get("attraction_name", ""),
        "route_name":        meta.get("route_name", ""),
        "hotel_name":        meta.get("hotel_name", ""),
        "restaurant_name":   meta.get("restaurant_name", ""),
        "region":            meta.get("region", ""),
        "source_file":       file_name,
        "source_path":       meta.get("source_path", ""),
    }

    if metadata["content_type"] not in VALID_CONTENT_TYPES:
        logger.warning(
            f"未知内容类型 '{metadata['content_type']}'，已重置为 '景点介绍'"
        )
        metadata["content_type"] = "景点介绍"

    logger.info(
        f"解析完成: {file_name} | 类型={metadata['content_type']} "
        f"| 正文长度={len(content)} 字符"
    )
    return content, metadata


def chunk_text(content: str, metadata: dict, chunk_size: int = 512, chunk_overlap: int = 64) -> list[dict]:
    """
    将正文切分为若干块，每块携带完整元数据和唯一 ID。

    chunk_size:    每块目标字符数（中文约 512 字）
    chunk_overlap: 相邻块重叠字符数（保持上下文连贯）

    返回: [{"id": str, "content": str, **metadata}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=MD_SEPARATORS,
        length_function=len,
    )
    texts = splitter.split_text(content)

    chunks = []
    for i, text in enumerate(texts):
        text = text.strip()
        if not text:
            continue
        chunk = {
            "id":      str(uuid.uuid4()),
            "content": text,
            **metadata,
            "chunk_index": i,
        }
        chunks.append(chunk)

    logger.info(f"分块完成: 共 {len(chunks)} 块，chunk_size={chunk_size}")
    return chunks
