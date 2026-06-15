# Travel RAG — 旅游智能知识库

基于 **RAG（检索增强生成）** 的旅游知识库问答系统。支持 Markdown 文档导入、混合向量检索、HyDE 扩展召回、Web 搜索、Reranker 重排序，以及 Qwen 流式问答。

## 功能特性

- **文档导入**：上传 `.md` 知识文档，自动解析、分块、向量化并写入 Milvus
- **混合检索**：稠密向量（text-embedding-v4）+ 稀疏向量（BGE-M3 / BM25 风格）
- **三路召回**：原始 Query 检索 + HyDE 假设文档检索 + Tavily Web 搜索（可选）
- **重排序**：BGE-Reranker-Large 精排 + 断崖截断
- **流式问答**：SSE 实时输出，前端展示「检索中 / 思考中」与引用来源
- **会话历史**：MongoDB 持久化多轮对话

## 技术栈

| 层级 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| 工作流 | LangGraph |
| LLM | 阿里云百炼 Qwen（qwen-plus） |
| Embedding | text-embedding-v4（1024 维） |
| 稀疏向量 / Reranker | BGE-M3、BGE-Reranker-Large（本地） |
| 向量库 | Milvus |
| 文档库 | MongoDB |
| 对象存储 | MinIO |
| Web 搜索 | Tavily（可选） |
| 前端 | HTML + 原生 JavaScript |

## 项目结构

```
travel/
├── main.py                 # FastAPI 入口
├── app/
│   ├── config.py           # 配置（Pydantic Settings）
│   ├── api/routes/         # ingest / chat / search / health
│   ├── processor/          # LangGraph：导入 / 检索 / RAG
│   └── utils/              # 向量、检索、Reranker、Web 搜索等
├── frontend/
│   └── index.html          # 单页前端（对话 + 导入文档）
├── docker-compose.yml      # Milvus / MongoDB / MinIO
├── requirements.txt
├── .env.example            # 环境变量模板（复制为 .env 后填写）
└── README.md
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/dinghaojie0515/travel.git
cd travel
```

### 2. 创建 Python 环境

```bash
conda create -n travel-rag python=3.10 -y
conda activate travel-rag
pip install -r requirements.txt
```

### 3. 启动 Docker 服务（Milvus / MongoDB / MinIO）

在服务器（如阿里云 Ubuntu）上：

```bash
docker compose up -d
```

### 4. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填写：

| 变量 | 说明 |
|------|------|
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key |
| `SERVER_IP` | Docker 服务所在服务器公网 IP |
| `BGE_M3_MODEL` | BGE-M3 本地路径或 HuggingFace 名 |
| `BGE_RERANKER_MODEL` | Reranker 本地路径或 HuggingFace 名 |
| `TAVILY_API_KEY` | （可选）启用 Web 搜索第三路召回 |

> **注意**：`.env` 含密钥与服务器信息，已在 `.gitignore` 中排除，请勿提交到 Git。

### 5. 下载本地模型（推荐）

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-m3', cache_dir='./models')"
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-reranker-large', cache_dir='./models')"
```

在 `.env` 中指向本地目录，例如：

```
BGE_M3_MODEL=./models/BAAI/bge-m3
BGE_RERANKER_MODEL=./models/BAAI/bge-reranker-large
```

### 6. 启动应用

```bash
python main.py
```

浏览器访问：`http://localhost:8000`

- API 文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/health/services`

## 核心流程

### 文档导入

```
上传 .md → 解析 → 分块 → 稀疏/稠密向量化 → Milvus 写入 → MinIO 归档
```

### 问答检索（RAG）

```
用户提问 → 加载历史 → 三路并发召回
  ├─ 原始 Query 混合检索（Milvus ANN + BM25，RRF 融合）
  ├─ HyDE 扩展检索
  └─ Tavily Web 搜索（可选）
→ 合并去重 → BGE-Reranker 重排 → 构建 Prompt → Qwen 流式生成 → 引用来源展示
```

## 主要 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/ingest/upload` | 上传 Markdown 文档 |
| GET | `/api/ingest/status/{task_id}` | 查询导入进度 |
| POST | `/api/chat/stream` | 流式问答（SSE） |
| POST | `/api/chat/` | 非流式问答 |
| GET | `/api/chat/history/{session_id}` | 会话历史 |
| GET | `/api/search/` | 知识检索 |

## 文档元数据（可选）

在 Markdown 文件头部使用 YAML frontmatter：

```yaml
---
content_type: 景点介绍
attraction_name: 天涯海角
region: 三亚
---
```

## 安全说明

以下内容**不会**也不应提交到仓库：

- `.env`（API Key、服务器 IP、Tavily Key）
- 本地模型权重目录
- 运行时日志与缓存

请仅提交 `.env.example` 作为配置模板。

## License

MIT
