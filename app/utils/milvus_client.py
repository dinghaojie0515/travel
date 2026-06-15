from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from loguru import logger
from app.config import settings


async def init_milvus():
    """初始化 Milvus 连接，并确保 Collection 存在"""
    try:
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
        )
        logger.info(f"Milvus 连接成功: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        _ensure_collection()
    except Exception as e:
        logger.error(f"Milvus 连接失败: {e}")
        raise


def _ensure_collection():
    """如果 Collection 不存在则创建；如果维度不匹配则先删除再重建"""
    col_name = settings.MILVUS_COLLECTION
    if utility.has_collection(col_name):
        # 检查已有 Collection 的向量维度是否和配置一致
        existing = Collection(col_name)
        for field in existing.schema.fields:
            if field.name == "dense_vector":
                existing_dim = field.params.get("dim", -1)
                if existing_dim != settings.EMBEDDING_DIM:
                    logger.warning(
                        f"Collection '{col_name}' 维度不匹配 "
                        f"(已有={existing_dim}, 配置={settings.EMBEDDING_DIM})，自动重建..."
                    )
                    utility.drop_collection(col_name)
                    break
                else:
                    logger.info(f"Milvus Collection '{col_name}' 已存在，维度={existing_dim}")
                    return

    logger.info(f"创建 Milvus Collection '{col_name}'...")
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="attraction_name", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="route_name", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="hotel_name", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="restaurant_name", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
        # 稠密向量（text-embedding-v4，维度由 EMBEDDING_DIM 配置）
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIM),
        # 稀疏向量（BGE-M3 BM25，用于混合检索）
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    schema = CollectionSchema(fields=fields, description="旅游知识库")
    collection = Collection(name=col_name, schema=schema)

    # 为稠密向量创建 IVF_FLAT 索引
    collection.create_index(
        field_name="dense_vector",
        index_params={"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 1024}},
    )
    # 为稀疏向量创建 SPARSE_INVERTED_INDEX 索引
    collection.create_index(
        field_name="sparse_vector",
        index_params={"metric_type": "IP", "index_type": "SPARSE_INVERTED_INDEX", "params": {"drop_ratio_build": 0.2}},
    )
    collection.load()
    logger.info(f"Milvus Collection '{col_name}' 创建完成")


def get_collection() -> Collection:
    """获取已加载的 Collection 实例"""
    col = Collection(settings.MILVUS_COLLECTION)
    col.load()
    return col
