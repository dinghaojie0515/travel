# Travel RAG — 旅游智能知识库

基于 **RAG（检索增强生成）** 的旅游知识问答系统。支持 Markdown 文档导入、混合向量检索、HyDE 扩展召回、Web 搜索、Reranker 重排序，以及流式问答与引用来源展示。

## 功能特性

- **文档导入**：上传 `.md` 知识文档，自动完成解析、分块、向量化并写入 Milvus
- **混合检索**：稠密向量（text-embedding-v4）+ 稀疏向量（BGE-M3 / BM25 风格）RRF 融合
- **三路召回**：原始 Query 检索 + HyDE 假设文档检索 + Tavily Web 搜索（可选）
- **重排序**：BGE-Reranker-Large 精排 + 断崖截断
- **流式问答**：SSE 实时输出，前端展示「检索中 / 思考中」状态与引用来源
- **会话历史**：MongoDB 持久化多轮对话

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | FastAPI + Uvicorn |
| 工作流 | LangGraph |
| 大模型 | 阿里云百炼 Qwen（qwen-plus） |
| 向量库 | Milvus |
| 文档库 | MongoDB |
| 对象存储 | MinIO |
| 本地模型 | BGE-M3、BGE-Reranker-Large |
| 前端 | HTML + CSS + JavaScript（marked.js） |

## 项目结构

```
travel/
├── main.py                 # FastAPI 入口
├── app/
│   ├── config.py           # 配置（Pydantic Settings）
│   ├── api/routes/         # HTTP 接口
│   ├── processor/          # LangGraph 流水线（导入 / 检索 / RAG）
│   └── utils/              # 向量、检索、LLM、存储等工具
├── frontend/
│   └── index.html          # 前端单页（智能对话 + 导入文档）
├── docker-compose.yml      # Milvus / MongoDB / MinIO
├── requirements.txt
└── .env.example            # 环境变量模板（复制为 .env 后填写）
```

## 架构流程

### 文档导入

```
上传 .md → 解析 Markdown → 智能分块 → 稀疏/稠密向量化 → Milvus 写入 → MinIO 归档
```

### RAG 问答

```
用户提问 → 加载历史 → 三路并发召回（向量 + HyDE + Web）
         → 合并去重 → Reranker 重排 → 构建 Prompt → Qwen 流式生成 → 保存历史
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

### 3. 启动基础服务（Docker）

在服务器上执行：

```bash
docker compose up -d
```

服务包括：Milvus（19530）、MongoDB（27017）、MinIO（9000/9001）。

### 4. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填写：

| 变量 | 说明 |
|------|------|
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key |
| `SERVER_IP` | Docker 服务所在服务器公网 IP |
| `BGE_M3_MODEL` | 本地 BGE-M3 模型路径（可选，默认 HuggingFace 名） |
| `BGE_RERANKER_MODEL` | 本地 Reranker 模型路径（可选） |
| `TAVILY_API_KEY` | Tavily Web 搜索 Key（可选，启用第三路召回） |

> **注意**：`.env` 已被 `.gitignore` 忽略，请勿将密钥提交到 Git。

### 5. 下载本地模型（推荐）

国内可使用 ModelScope：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-m3', cache_dir='./models')"
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-reranker-large', cache_dir='./models')"
```

然后在 `.env` 中指向本地路径，例如：

```
BGE_M3_MODEL=./models/bge-m3
BGE_RERANKER_MODEL=./models/bge-reranker-large
```

### 6. 启动应用

```bash
python main.py
```

访问：

- 前端界面：http://localhost:8000
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health/services

## 主要 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/ingest/upload` | 上传 Markdown 文档 |
| GET | `/api/ingest/status/{task_id}` | 查询导入进度 |
| POST | `/api/chat/stream` | 流式问答（SSE） |
| POST | `/api/chat/` | 非流式问答 |
| GET | `/api/search/` | 知识检索 |
| GET | `/health/services` | 服务健康检查 |

## 文档格式

支持 YAML frontmatter 元数据：

```yaml
---
content_type: 景点介绍
attraction_name: 天涯海角
region: 三亚
---

正文内容...
```

## License

MIT
