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

## Stage 2: ReAct Agent（已完成）

**完成日期:** 2026-05-05
**提交:** `56c6d6e`

### 核心概念

#### 1. ReAct 模式 (Reasoning + Acting)

ReAct = Thought → Action → Observation 循环，让 Agent 能像人类一样边推理边行动。

```
用户问题
  ↓
Thought: 我需要查什么？
  ↓
Action: search_knowledge_base
  ↓
Observation: 检索结果
  ↓
Thought: 这个答案够了吗？
  ↓
Action: 再查一次 / 直接回答
  ↓
...循环直到完成
  ↓
Final Answer: 回答
```

#### 2. StructuredTool vs @tool 装饰器

LangChain 创建 Tool 的两种方式：

```python
# 方式 1: @tool 装饰器（简单场景）
from langchain_core.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """检索知识库内容"""
    return vector_store.search(query)

# 方式 2: StructuredTool（推荐，更明确）
from langchain_core.tools import StructuredTool

def search_func(query: str) -> str:
    return vector_store.search(query)

tool = StructuredTool(
    name="search_knowledge_base",
    description="检索知识库文档内容",
    func=search_func,
    args_schema={"query": {"type": "string", "description": "用户问题"}}
)
```

**推荐使用 StructuredTool**，因为：
- 参数类型更明确
- 更容易做验证
- 文档更清晰

#### 3. create_agent (新 API)

LangChain 1.2+ 使用 `create_agent` 替代旧的 `create_react_agent`：

```python
# 旧 API (可能已废弃)
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 新 API
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)
result = agent.invoke({"input": question}, config=RunnableConfig(max_iterations=5))
```

#### 4. RunnableConfig

配置 Agent 执行参数：

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    max_iterations=5,  # 最多推理步数
    verbose=True       # 显示推理过程
)
result = agent.invoke(input_dict, config=config)
```

#### 5. Agent 的输入输出结构

**输入:**
```python
{
    "input": "用户问题",
    "chat_history": "对话历史（格式化字符串）",
    "agent_scratchpad": []  # 中间推理过程
}
```

**输出:**
```python
{
    "output": "最终回答",
    "messages": [AIMessage, ToolMessage, ...]  # 完整消息历史
}
```

### 关键代码模式

```python
# 1. 创建 Tool
class KnowledgeBaseSearchTool:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def search(self, query: str) -> str:
        docs = self.vector_store.similarity_search(query, k=4)
        return "\n\n".join([doc.page_content for doc in docs])

tool = StructuredTool(
    name="search_knowledge_base",
    func=search_func,
    description="检索知识库...",
    args_schema={"query": {"type": "string"}}
)

# 2. 创建 Agent
agent = create_agent(
    model=llm,
    tools=[tool],
    system_prompt="你是一个助手..."
)

# 3. 调用 Agent
result = agent.invoke(
    {"input": question, "chat_history": ""},
    config=RunnableConfig(max_iterations=5, verbose=True)
)
```

### 面试要点

1. **ReAct 是什么?** - Reasoning + Acting 循环，Thought → Action → Observation 迭代
2. **什么时候用 ReAct?** - 需要多步推理、工具调用、动态规划的复杂任务
3. **StructuredTool 和 @tool 的区别?** - StructuredTool 更正式，推荐用于生产环境
4. **max_iterations 用来做什么?** - 限制 Agent 最大推理步数，防止无限循环
5. **chat_history 为什么是字符串?** - Agent 需要将历史对话作为上下文传入

## Stage 3: Reflection 机制（已完成）

**完成日期:** 2026-05-05
**提交:** `cbfb4ee`

### 核心概念

#### 1. Self-Reflection 机制

Self-Reflection 是 Agent 自我反思的能力，让 Agent 能检查自己的输出质量并在必要时修正。

```
Draft Answer → Quality Check → [Passed] → Final Answer
                           ↓
                      [Failed] → Refine → Final Answer
```

#### 2. QualityChecker - 回答质量检查器

检查回答的三个维度：
- 回答是否基于提供的文档？
- 是否有事实性错误？
- 是否完整回答了问题？

```python
class QualityChecker:
    def __init__(self, llm):
        self.chain = prompt | llm | StrOutputParser()

    def check(self, question, answer, context) -> dict:
        result = self.chain.invoke({
            "question": question,
            "answer": answer,
            "context": context
        })
        # 解析返回的 Quality: passed/failed, Reason, Suggestion
```

#### 3. ReflectionAgent - 带反思的 Agent

```python
class ReflectionAgent:
    def process_with_reflection(self, question, draft_answer, context):
        # 1. 检查质量
        quality = self.checker.check(question, draft_answer, context)

        # 2. 不合格则修正
        if quality.get("quality") == "failed":
            refined = self.refine_answer(question, draft_answer, context)
            return {"answer": refined, "was_refined": True}

        return {"answer": draft_answer, "was_refined": False}
```

#### 4. 修正循环 (Refinement Loop)

```python
# 最多修正 max_refinement 次
for i in range(self.max_refinement):
    quality = self.checker.check(question, draft_answer, context)
    if quality.get("quality") == "passed":
        break
    draft_answer = self.refine_answer(question, draft_answer, context)
```

### 关键代码模式

```python
# 1. 质量检查 Chain
quality_prompt = ChatPromptTemplate.from_template("""
检查以下回答的质量：
问题：{question}
回答：{answer}
文档内容：{context}

输出格式：
Quality: [passed/failed]
Reason: [原因]
Suggestion: [改进建议]
""")
quality_chain = quality_prompt | llm | StrOutputParser()

# 2. 修正 Chain
refine_prompt = ChatPromptTemplate.from_template("""
根据建议改进回答：
问题：{question}
原始回答：{draft_answer}
改进建议：{suggestion}
""")
refine_chain = refine_prompt | llm | StrOutputParser()
```

### 面试要点

1. **Self-Reflection 是什么?** - Agent 检查自己输出的质量并在必要时修正的能力
2. **为什么需要 Reflection?** - 提高回答准确性，避免错误传播
3. **QualityChecker 检查哪些维度?** - 基于文档、事实准确性、回答完整性
4. **Refinement Loop 如何工作?** - 检查 → 不合格则修正 → 再次检查 → 最多 N 次

## Stage 4: Memory 升级（已完成）

**完成日期:** 2026-05-05
**提交:** `6f35f78`

### 核心概念

#### 1. LangChain Memory 组件

LangChain 提供了统一的 Memory 接口，用于管理对话历史。

```
Redis (持久化) ←→ ConversationMemory ←→ LangChain Memory 接口
```

#### 2. ChatMessageHistory

LangChain 的内存历史类，存储消息列表：

```python
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()
history.add_user_message("你好")
history.add_ai_message("你好！有什么可以帮助你的？")
```

#### 3. to_langchain_history()

将 Redis 中的会话历史转换为 LangChain 的 ChatMessageHistory：

```python
def to_langchain_history(self, session_id: str):
    from langchain.memory import ChatMessageHistory as LCChatMessageHistory

    history = self.get_full_history(session_id)
    chat_history = LCChatMessageHistory()

    for msg in history:
        if msg["role"] == "user":
            chat_history.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            chat_history.add_ai_message(msg["content"])

    return chat_history
```

#### 4. get_langchain_messages()

获取 LangChain 格式的消息列表（用于直接传入 Prompt）：

```python
def get_langchain_messages(self, session_id: str) -> list:
    from langchain_core.messages import HumanMessage, AIMessage

    history = self.get_full_history(session_id)
    messages = []

    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    return messages
```

#### 5. RunnableWithMessageHistory

LangChain 提供的 Runnable 带历史功能：

```python
from langchain_core.runnables import RunnableWithMessageHistory

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # 函数，返回 ChatMessageHistory
    input_messages_key="input"
)
```

### 关键代码模式

```python
# 1. 转换为 LangChain 历史
chat_history = memory.to_langchain_history(session_id)

# 2. 获取消息列表直接用于 Prompt
messages = memory.get_langchain_messages(session_id)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手"),
    *messages  # 解包消息列表
])
```

### 面试要点

1. **LangChain Memory 是什么?** - LangChain 提供的统一 Memory 接口，用于管理对话历史
2. **ChatMessageHistory 的作用?** - 存储对话消息，支持 add_user_message/add_ai_message
3. **为什么要适配 LangChain?** - 让现有系统能接入 LangChain 生态，使用 RunnableWithMessageHistory 等组件
4. **get_langchain_messages 和 to_langchain_history 的区别?** - 前者返回消息对象列表，后者返回完整的 ChatMessageHistory 对象

---

## 学习进度

| Stage | 状态 | 完成日期 | 提交 |
|-------|------|----------|------|
| Stage 1: LangChain 基础 | ✅ 完成 | 2026-05-05 | `3f0f9fd` |
| Stage 2: ReAct Agent | ✅ 完成 | 2026-05-05 | `56c6d6e` |
| Stage 3: Reflection | ✅ 完成 | 2026-05-05 | `cbfb4ee` |
| Stage 4: Memory 升级 | ✅ 完成 | 2026-05-05 | `6f35f78` |