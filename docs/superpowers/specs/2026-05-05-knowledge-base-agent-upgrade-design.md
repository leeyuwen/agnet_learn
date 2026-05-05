# Knowledge Base QA Agent 升级设计

## 目标

将现有的固定 RAG 流程升级为基于 LangChain 的推理型 Agent，支持：
- ReAct 推理+行动循环
- Reflection 自我反思机制
- 统一的 Memory 管理

## 现状分析

### 当前架构

```
用户问题 → 向量检索 → 拼接 Prompt → LLM → 回答
```

**问题：**
- 单次检索，无推理循环
- 无法判断检索结果是否足够
- 无自我修正能力

### 目标架构

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
┌─────────────────────────────┐
│  Reflection: 检查回答质量    │
│  不合格则修正               │
└─────────────────────────────┘
    ↓
最终回答 + 参考来源
```

## 分阶段实现

### Stage 1: LangChain 基础（1-2周）

**目标：** 理解 LCEL，用 LangChain 方式调用 LLM

**文件：** 新增 `stage1_basics.py`

**内容：**
- PromptTemplate / ChatPromptTemplate
- LLM Chain 串联
- 输出解析（StrOutputParser）

**练习：**
用 LCEL 重写 agent.py 中的 LLM 调用部分

```python
# 目标：替代直接调用的方式
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "...", "context": "..."})
```

---

### Stage 2: ReAct Agent（2-3周）

**目标：** 实现推理+行动循环

**文件：** 修改 `agent.py`

**核心组件：**

1. **Tool 定义**
```python
from langchain_core.tools import tool

@tool
def search_knowledge_base(query: str) -> list:
    """检索知识库内容。当用户询问文档相关问题时使用。"""
    # 调用 vector_store.similarity_search
```

2. **ReAct Agent 创建**
```python
from langchain.agents import create_react_agent

agent = create_react_agent(llm, tools, prompt)
result = agent.invoke({"input": user_question})
```

3. **中间步骤输出**
```python
# 查看 Agent 的推理过程
for step in result["intermediate_steps"]:
    print(f"Action: {step[0].tool}")
    print(f"Input: {step[0].tool_input}")
    print(f"Observation: {step[1]}")
```

**ReAct Prompt 模板：**
```
你是一个知识库问答助手。你可以使用工具来检索信息。

常用工具：
- search_knowledge_base: 检索知识库文档

当你收到问题：
1. 思考需要查什么
2. 调用合适的工具
3. 根据结果判断是否需要继续检索
4. 最终给出回答

输出格式：
Thought: [你的思考]
Action: [工具名称]
Action Input: [输入]
Observation: [工具返回结果]
...（可多次循环）
Final Answer: [最终回答]
```

---

### Stage 3: Reflection 机制（2周）

**目标：** 自我反思，回答质量检测与修正

**文件：** 新增 `reflection_agent.py`

**核心逻辑：**

```python
def reflection_loop(question, draft_answer, context):
    """检查回答质量，不合格则修正"""

    quality_check_prompt = f"""
    检查以下回答的质量：

    问题：{question}
    回答：{draft_answer}
    文档内容：{context}

    检查项：
    1. 回答是否基于提供的文档？
    2. 是否有事实性错误？
    3. 是否完整回答了问题？

    如果回答不合格，说明需要修正什么。
    """

    quality = llm.invoke(quality_check_prompt)

    if "不合格" in quality:
        # 修正回答
        revised = llm.invoke(f"修正回答：{draft_answer}，问题：{question}")
        return revised

    return draft_answer
```

---

### Stage 4: Memory 升级（1-2周）

**目标：** 对接 LangChain Memory 组件

**文件：** 修改 `conversation_memory.py`

**核心组件：**

```python
from langchain.memory import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

# 改造 conversation_memory
class ConversationMemory:
    # 保留 Redis 存储
    # 增加 to_langchain_history() 方法

    def to_langchain_history(self, session_id: str) -> ChatMessageHistory:
        """转换为 LangChain 的 ChatMessageHistory"""
        history = self.get_full_history(session_id)
        chat_history = ChatMessageHistory()
        for msg in history:
            if msg["role"] == "user":
                chat_history.add_user_message(msg["content"])
            else:
                chat_history.add_ai_message(msg["content"])
        return chat_history
```

**RunnableWithMessageHistory 集成：**
```python
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # 从 Redis 获取历史
    input_messages_key="input"
)
```

---

## 文件改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `stage1_basics.py` | 新增 | LangChain 基础练习 |
| `agent.py` | 修改 | 升级为 ReAct Agent |
| `reflection_agent.py` | 新增 | 自我反思机制 |
| `conversation_memory.py` | 修改 | 增加 LangChain 适配方法 |
| `main.py` | 修改 | 支持新 Agent |
| `docs/` | 更新 | 更新 README 说明新架构 |

## 验证标准

每个阶段完成后，需要能演示：

1. **Stage 1**：用 LCEL Chain 处理一个问答
2. **Stage 2**：Agent 能展示多步检索过程（可以看到 Thought/Action/Observation）
3. **Stage 3**：故意提问一个模糊问题，Agent 能追问或clarify
4. **Stage 4**：跨会话记忆测试

## 学习资源

- LangChain Quickstart: https://python.langchain.com/docs/get_started
- ReAct Agent: https://python.langchain.com/docs/how_to/react_agent
- LangChain Memory: https://python.langchain.com/docs/modules/memory

## 时间安排

| 周次 | 阶段 | 内容 |
|------|------|------|
| 1-2 | Stage 1 | LangChain 基础、LCEL |
| 3-5 | Stage 2 | ReAct Agent 实现 |
| 6-7 | Stage 3 | Reflection 机制 |
| 8-9 | Stage 4 | Memory 升级 |

---

## 风险与注意事项

1. **MiniMax API 兼容性**：确保 LangChain 的 ChatOpenAI 兼容 MiniMax 的 API 格式
2. **Redis 依赖**：继续使用 Redis 存储会话历史，保持数据持久化
3. **向量检索保持不变**：Chroma 作为 Tool 被 Agent 调用，底层不变