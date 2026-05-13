# 知识库问答系统 (Knowledge Base QA) - ReAct Agent 版本

## 项目概述

基于 ReAct (Reasoning + Acting) 模式的知识库问答系统，支持：
- 多步推理检索
- 自我反思与回答修正
- 统一的会话记忆管理

## 架构图

```
用户问题
    ↓
┌─────────────────────────────┐
│     ReAct AgentExecutor      │
│  ┌─────────────────────────┐│
│  │  Thought: 需要查什么？   ││
│  │  Action: 检索工具        ││
│  │  Observation: 结果       ││
│  │  (循环直到完成)          ││
│  └─────────────────────────┘│
└─────────────────────────────┘
    ↓
最终回答 + 参考来源
```

## 文件结构

| 文件 | 说明 |
|------|------|
| `agent.py` | ReAct Agent 实现，支持多步推理 |
| `stage1_basics.py` | LangChain 基础练习（LCEL） |
| `reflection_agent.py` | 自我反思机制 |
| `conversation_memory.py` | 会话记忆（支持 LangChain Memory） |
| `vector_store.py` | 向量检索（Chroma） |
| `main.py` | 入口程序 |

## 各 Stage 说明

### Stage 1: LangChain 基础
- LCEL (LangChain Expression Language) 语法
- PromptTemplate / ChatPromptTemplate
- Chain 串联

### Stage 2: ReAct Agent
- create_agent 创建推理型 Agent
- StructuredTool 定义工具
- AgentExecutor 多步推理循环

### Stage 3: Reflection
- QualityChecker 回答质量检查
- ReflectionAgent 自我反思
- 回答修正流程

### Stage 4: Memory 升级
- to_langchain_history() 方法
- LangChain ChatMessageHistory 适配
- RunnableWithMessageHistory 集成支持

## 环境配置

### 1. 配置 .env 文件

创建 `knowledge_base_qa/.env` 文件：
```
MINIMAX_API_KEY=your_api_key
MINIMAX_MODEL=your_model
HF_TOKEN=your_hf_token
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 2. 启动 Redis（必须）

Redis 用于会话记忆存储，必须先启动。

**方法一：使用 Docker（推荐）**
```bash
# 启动 Redis 容器
docker run -d -p 6379:6379 redis

# 验证 Redis 运行状态
docker ps

# 测试连接
redis-cli ping
```

**方法二：Windows 安装 Redis**
```powershell
# 使用 Chocolatey 安装
choco install redis-64 -y

# 启动 Redis 服务
redis-server
```

**方法三：WSL2 安装 Redis（如果您已安装 WSL）**
```bash
sudo apt update
sudo apt install redis-server
sudo service redis-server start
```

**常见问题**
- 如果遇到 "由于目标计算机积极拒绝，无法连接" 错误，说明 Redis 未启动
- 使用 `redis-cli ping` 验证 Redis 是否正常运行（返回 PONG 表示成功）

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
# 初始化知识库
python main.py --init

# 启动问答（显示推理过程）
python main.py
```
