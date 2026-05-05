# AI Agent 学习笔记 - 实战记录

本文档记录从固定 RAG 升级为推理型 Agent 的学习过程。

---

## Stage 1: LangChain 基础（LCEL）

**完成日期:** 2026-05-05
**提交:** `3f0f9fd`

### 核心概念

#### 1. LCEL (LangChain Expression Language)

LCEL 是 LangChain 的链式表达式语法，通过 `|` 操作符串联各个组件。

```python
chain = prompt | llm | output_parser
```

**为什么重要:** LCEL 让 LLM 应用的构建像搭积木一样简单，每个组件职责单一，通过 pipe 连接。

#### 2. PromptTemplate vs ChatPromptTemplate

| 类 | 用途 | 场景 |
|---|------|------|
| `PromptTemplate` | 纯文本模板 | 简单文本生成 |
| `ChatPromptTemplate` | 消息模板 | 对话场景（包含 system/human/assistant 角色） |

```python
# 文本模板
prompt = PromptTemplate.from_template("解释 {concept}")

# 聊天模板（推荐）
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手"),
    ("human", "{question}")
])
```

#### 3. Chain 的串联方式

```
prompt (PromptTemplate) → llm (ChatOpenAI) → output_parser (StrOutputParser)
```

**示例:**
```python
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("用一句话解释: {concept}")
    | ChatOpenAI(model="gpt-4")
    | StrOutputParser()
)

result = chain.invoke({"concept": "向量数据库"})
```

#### 4. StrOutputParser

将 LLM 输出解析为字符串（而不是 AIMessage 对象）。

**对比:**
```python
# 不使用 output_parser
response = llm.invoke(prompt)  # 返回 AIMessage 对象

# 使用 output_parser
response = chain.invoke(...)  # 返回 str
```

#### 5. 工厂函数模式

创建可复用的 Chain 实例：

```python
def create_qa_chain():
    """工厂函数：每次调用创建新的 Chain 实例"""
    basics = LangChainBasics()
    return basics.create_qa_chain()
```

### 关键代码模式

```python
# 1. 初始化 LLM
llm = ChatOpenAI(
    api_key=Config.MINIMAX_API_KEY,
    base_url=Config.MINIMAX_BASE_URL,
    model_name=Config.MINIMAX_MODEL,
    temperature=0.7
)

# 2. 创建 Prompt
prompt = ChatPromptTemplate.from_template("问题: {question}\n上下文: {context}")

# 3. 串联 Chain
chain = prompt | llm | StrOutputParser()

# 4. 调用
result = chain.invoke({"question": "...", "context": "..."})
```

### 面试要点

1. **LCEL 是什么?** - LangChain 的链式表达式语法，通过 `|` 操作符串联 prompt、llm、output_parser
2. **LCEL 的优势?** - 声明式写法，组件可插拔，调试方便
3. **什么时候用 ChatPromptTemplate?** - 对话场景，需要区分 system/human/assistant 角色时
4. **OutputParser 的作用?** - 将 LLM 的原始输出（可能是 AIMessage 对象）转换为结构化数据（字符串、JSON 等）

---

## Stage 2: ReAct Agent（待完成）

## Stage 3: Reflection 机制（待完成）

## Stage 4: Memory 升级（待完成）

---

## 学习进度

| Stage | 状态 | 完成日期 | 提交 |
|-------|------|----------|------|
| Stage 1: LangChain 基础 | ✅ 完成 | 2026-05-05 | `3f0f9fd` |
| Stage 2: ReAct Agent | ⏳ 待开始 | - | - |
| Stage 3: Reflection | ⏳ 待开始 | - | - |
| Stage 4: Memory 升级 | ⏳ 待开始 | - | - |