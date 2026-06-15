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
├── docker-compose.yml      # Milvus / MongoDB / MinIO 服务
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
pip install tavily-python   # 可选，启用 Web 搜索
```

### 3. 启动 Docker 服务

在服务器（或本地）启动 Milvus、MongoDB、MinIO：

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
| `SERVER_IP` | Docker 服务所在机器 IP（本地开发填 `127.0.0.1`） |
| `BGE_M3_MODEL` | BGE-M3 本地路径或 HuggingFace 模型名 |
| `BGE_RERANKER_MODEL` | Reranker 本地路径或模型名 |
| `TAVILY_API_KEY` | 可选，启用 Web 搜索第三路召回 |

> **注意**：`.env` 已加入 `.gitignore`，请勿将密钥提交到 Git。

### 5. 下载本地模型（推荐）

国内可使用 ModelScope：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-m3', cache_dir='./models')"
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-reranker-large', cache_dir='./models')"
```

然后在 `.env` 中指向本地目录，例如：

```
BGE_M3_MODEL=./models/bge-m3
BGE_RERANKER_MODEL=./models/bge-reranker-large
```

### 6. 启动应用

```bash
python main.py
```

浏览器访问：`http://localhost:8000`

- **智能对话**：顶部导航「智能对话」
- **导入文档**：顶部导航「导入文档」，上传 `.md` 文件
- **API 文档**：`http://localhost:8000/docs`

## 核心流程

### 文档导入

```
上传 .md → 解析 YAML 元数据 → 文本分块 → 稀疏/稠密向量化 → Milvus 写入 → MinIO 归档
```

### RAG 问答

```
用户提问 → 加载历史 → 三路并发召回（向量 + HyDE + Web）
         → 合并去重 → Reranker 重排 → 构建 Prompt → Qwen 流式生成 → 展示引用来源
```

## API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/ingest/upload` | 上传 Markdown 文档 |
| GET | `/api/ingest/status/{task_id}` | 查询导入进度 |
| POST | `/api/chat/stream` | 流式问答（SSE） |
| POST | `/api/chat/` | 非流式问答 |
| GET | `/api/chat/history/{session_id}` | 会话历史 |
| GET | `/api/search/` | 知识检索 |
| GET | `/health/services` | 服务健康检查 |

## 文档元数据（可选）

在 `.md` 文件开头添加 YAML frontmatter，可丰富引用来源信息：

```yaml
---
content_type: 景点介绍
attraction_name: 天涯海角
region: 三亚
---
```

## 开发说明

- **本地开发**：Python 在 Windows/Mac 本地运行，Docker 服务可部署在阿里云 ECS
- **向量维度**：`text-embedding-v4` 默认 1024 维，请保持 `EMBEDDING_DIM=1024`
- **pymilvus**：需 `>=2.5.0`

## 许可证

MIT
