from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 阿里云百炼
    DASHSCOPE_API_KEY: str
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL: str = "qwen-plus"
    EMBEDDING_MODEL: str = "text-embedding-v4"
    EMBEDDING_DIM: int = 1024  # text-embedding-v4 默认返回 1024 维

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "travel_knowledge"

    # MongoDB
    MONGODB_URI: str = "mongodb://admin:admin123@localhost:27017"
    MONGODB_DB: str = "travel_rag"
    MONGODB_COLLECTION_HISTORY: str = "chat_history"

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin123"
    MINIO_BUCKET: str = "travel-docs"
    MINIO_SECURE: bool = False

    # 本地模型
    BGE_M3_MODEL: str = "BAAI/bge-m3"
    BGE_RERANKER_MODEL: str = "BAAI/bge-reranker-large"

    # Web 搜索（可选，填写后启用第三路召回）
    TAVILY_API_KEY: str = ""
    WEB_SEARCH_MAX_RESULTS: int = 5   # 每次搜索返回最多几条结果

    # FastAPI
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_RELOAD: bool = True


settings = Settings()
