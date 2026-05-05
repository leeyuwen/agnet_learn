# 知识库问答系统 (Knowledge Base QA)

## 项目概述

基于 RAG（检索增强生成）模式的知识库问答系统，支持文档检索和对话记忆。

## 架构图

```
用户输入问题
      ↓
┌─────────────────┐
│    agent.py      │  ← 核心智能体
│  KnowledgeBase   │
│    Agent         │
└────────┬────────┘
         │
    ┌────┴────┬────────────┐
    ↓         ↓            ↓
┌───────┐ ┌────────┐ ┌───────────┐
│向量检索│ │会话记忆│ │   LLM    │
│ (向量库)│ │ (Redis)│ │ (MiniMax)│
└───────┘ └────────┘ └───────────┘
    ↑         ↑
    │         │
┌────────┐ ┌────────────┐
│文档加载│ │  向量存储  │
│document│ │vector_store│
│_loader │ │ (Chroma)   │
└────────┘ └────────────┘
```

## 模块说明

### 1. config.py - 配置管理

集中管理所有配置项，从 `.env` 文件读取环境变量。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `MINIMAX_API_KEY` | MiniMax API 密钥 | - |
| `MINIMAX_BASE_URL` | API 地址 | `https://api.minimax.chat/v1` |
| `MINIMAX_MODEL` | 模型名称 | - |
| `SESSION_TTL` | 会话过期时间 | 86400秒（1天） |
| `CHUNK_SIZE` | 文档切割块大小 | 500字符 |
| `CHUNK_OVERLAP` | 切割重叠大小 | 50字符 |
| `DOCS_FOLDER` | 文档存放文件夹 | `knowledge_base_qa/docs` |

### 2. document_loader.py - 文档加载器

负责加载和切割文档。

**支持格式：**
- `.txt` - 文本文件
- `.pdf` - PDF 文件
- `.md` - Markdown 文件

**核心流程：**
```
load_folder()     → 扫描文件夹，加载所有文档
        ↓
load_single_file() → 根据扩展名选择加载器
        ↓
split_documents() → RecursiveCharacterTextSplitter 切割
        ↓
返回 Document[] 列表
```

**关键代码：**
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每块500字符
    chunk_overlap=50     # 块之间重叠50字符
)
```

### 3. vector_store.py - 向量存储

基于 Chroma 向量数据库，实现语义检索。

**核心组件：**
- **Embedding 模型**：`shibing624/text2vec-base-chinese`（中文文本转向量）
- **向量数据库**：Chroma（持久化存储）
- **相似度度量**：余弦相似度（cosine）

**核心方法：**

| 方法 | 功能 |
|------|------|
| `create_vector_store(documents)` | 创建向量索引 |
| `load_vector_store()` | 加载已有向量库 |
| `similarity_search(query, k=4)` | 检索最相似的 k 个文档块 |
| `clear_index()` | 清空向量索引 |
| `add_documents(documents)` | 添加文档到向量库 |

**检索流程：**
```python
# 1. 用户问题转为向量
# 2. 在向量库中搜索最相似的文档块
# 3. 返回 top-k 个结果
```

### 4. conversation_memory.py - 会话记忆

基于 Redis 存储对话历史。

**数据结构：**
```python
{
    "role": "user/assistant",
    "content": "消息内容",
    "timestamp": "2024-01-01T00:00:00"
}
```

**存储方式：**
- Redis List，每个 session 一个 key
- `rpush` 追加消息到列表末尾
- 每次写入刷新 TTL（滑动过期）

**核心方法：**

| 方法 | 功能 |
|------|------|
| `add_message(session_id, role, content)` | 添加消息 |
| `get_history(session_id, limit=10)` | 获取最近 N 条 |
| `get_full_history(session_id)` | 获取全部历史 |
| `clear_session(session_id)` | 清除会话 |
| `format_for_llm(session_id)` | 格式化为字符串供 LLM 使用 |

### 5. agent.py - 智能体核心

编排各个模块，协调工作流程。

**初始化：**
```python
self.vector_store = VectorStore()        # 向量检索
self.memory = ConversationMemory()       # 会话记忆
self.llm = ChatOpenAI(...)               # LLM 调用
self.tools = [知识库检索工具]             # 工具列表
```

**query() 流程：**
```
1. 保存用户消息到 Redis
        ↓
2. 获取对话历史（格式化为字符串）
        ↓
3. 检索相关文档（向量相似度搜索）
        ↓
4. 构建 Prompt：
   - System Message（角色设定 + 对话历史）
   - User Message（文档内容 + 问题）
        ↓
5. 调用 LLM 获取回答
        ↓
6. 保存助手回答到 Redis
        ↓
7. 返回回答 + 参考文档来源
```

### 6. main.py - 入口文件

提供 CLI 界面运行。

**两种运行模式：**

| 命令 | 行为 |
|------|------|
| `python main.py` | 初始化知识库 + 进入聊天循环 |
| `python main.py --init` | 仅初始化知识库 |

**聊天命令：**
- `quit` / `exit` - 退出
- `clear` - 清除当前会话历史

## RAG 工作流程

```
用户问题: "这个文档讲了什么？"
      ↓
┌─────────────────────────────────────────┐
│  1. 向量化问题                           │
│     text2vec("这个文档讲了什么？")        │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│  2. 向量检索                             │
│     Chroma similarity_search            │
│     → 返回最相关的4个文档块               │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│  3. 构建 Prompt                          │
│     [系统] 你是一个知识库问答助手         │
│     [文档] 相关段落1...                  │
│     [文档] 相关段落2...                  │
│     [问题] 这个文档讲了什么？             │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│  4. LLM 生成回答                         │
│     MiniMax API → "文档主要讲述了..."     │
└─────────────────────────────────────────┘
```

## 数据流

```
docs/ 文件夹
    ↓ document_loader
[Document, Document, ...]
    ↓ vector_store
Chroma 向量数据库 (chroma_db/)
    ↓ similarity_search
[Document, Document, ...] (最相关的 k 个)
    ↓ agent.query
Prompt → LLM → 回答
    ↓
conversation_memory (Redis)
```

## 环境依赖

```
langchain
langchain-community
langchain-openai
chromadb
redis
python-dotenv
```

## 使用步骤

1. **准备文档**
   ```
   knowledge_base_qa/docs/
   ├── README.md
   ├── guide.txt
   └── notes.pdf
   ```

2. **配置环境变量** (`.env`)
   ```
   MINIMAX_API_KEY=your_api_key
   MINIMAX_MODEL=your_model
   REDIS_HOST=localhost
   REDIS_PORT=6379
   ```

3. **初始化知识库**
   ```bash
   python main.py --init
   ```

4. **开始问答**
   ```bash
   python main.py
   ```

## 关键设计点

1. **滑动过期策略**：会话只要持续活跃就不会过期
2. **文档切割**：长文档切成 500 字符的小块，便于精确检索
3. **重叠切割**：相邻块重叠 50 字符，避免上下文断裂
4. **持久化向量库**：Chroma 将向量存储到磁盘，重启不丢失
