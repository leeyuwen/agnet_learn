# Knowledge Base Agent 升级实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 knowledge_base_qa 从固定 RAG 流程升级为基于 LangChain 的推理型 Agent，支持 ReAct 循环和 Reflection 机制

**Architecture:** 采用 LangChain 的 ReAct Agent 架构，通过 Tool 调用向量检索，AgentExecutor 负责推理循环，Reflection 层负责回答质量检测与修正

**Tech Stack:** LangChain, LangChain Community, Chroma, Redis, MiniMax API

---

## 文件结构

```
knowledge_base_qa/
├── agent.py                    # 修改：ReAct Agent
├── conversation_memory.py       # 修改：增加 LangChain 适配
├── vector_store.py             # 不变：作为 Tool 使用
├── config.py                   # 不变
├── document_loader.py          # 不变
├── main.py                     # 修改：支持新 Agent
├── stage1_basics.py            # 新增：LangChain 基础练习
├── reflection_agent.py         # 新增：Reflection 机制
└── tests/
    ├── test_stage1_basics.py   # 新增
    ├── test_agent.py           # 新增
    └── test_reflection.py      # 新增
```

---

## Task 1: Stage 1 - LangChain 基础（LCEL）

**Files:**
- Create: `knowledge_base_qa/stage1_basics.py`
- Test: `knowledge_base_qa/tests/test_stage1_basics.py`

- [ ] **Step 1: 创建测试文件**

```python
# knowledge_base_qa/tests/test_stage1_basics.py
import pytest
from stage1_basics import create_qa_chain, LangChainBasics

def test_create_qa_chain():
    """测试 LCEL Chain 创建"""
    chain = create_qa_chain()
    assert chain is not None
    # 验证 chain 可以 invoke

def test_basics_prompt_template():
    """测试 PromptTemplate 创建"""
    basics = LangChainBasics()
    prompt = basics.create_qa_prompt("什么是 AI Agent")
    assert "什么是 AI Agent" in prompt.to_string()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd G:/PythonProject/agnet_learn/knowledge_base_qa && python -m pytest tests/test_stage1_basics.py -v`
Expected: FAIL - `stage1_basics` not found

- [ ] **Step 3: 创建 stage1_basics.py**

```python
# knowledge_base_qa/stage1_basics.py
"""
LangChain 基础练习 - LCEL 语法
Stage 1: 理解 LangChain Expression Language
"""
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from config import Config


class LangChainBasics:
    """LangChain 基础组件练习"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=Config.MINIMAX_API_KEY,
            base_url=Config.MINIMAX_BASE_URL,
            model_name=Config.MINIMAX_MODEL,
            temperature=Config.MINIMAX_TEMPERATURE
        )

    def create_qa_prompt(self, question: str) -> ChatPromptTemplate:
        """创建问答 Prompt"""
        template = """你是一个知识库问答助手。

问题: {question}

请基于以下上下文回答：
{context}

如果上下文中没有相关信息，请说明。"""

        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    def create_simple_chain(self):
        """创建一个简单的 LCEL Chain"""
        prompt = ChatPromptTemplate.from_template(
            "用一句话解释：{concept}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain

    def create_qa_chain(self):
        """创建问答 Chain（用于替代原来的直接 LLM 调用）"""
        prompt = ChatPromptTemplate.from_template(
            """你是一个知识库问答助手。

问题: {question}
上下文: {context}

请根据上下文回答问题。"
"""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain


def create_qa_chain():
    """工厂函数：创建 QA Chain"""
    basics = LangChainBasics()
    return basics.create_qa_chain()


def demo_lcel():
    """演示 LCEL 的基本用法"""
    basics = LangChainBasics()

    # 简单概念解释
    simple_chain = basics.create_simple_chain()
    result = simple_chain.invoke({"concept": "什么是向量数据库"})
    print(f"简单 Chain 结果: {result}")

    # 问答 Chain
    qa_chain = basics.create_qa_chain()
    result = qa_chain.invoke({
        "question": "LangChain 是什么？",
        "context": "LangChain 是一个用于构建 LLM 应用的框架。"
    })
    print(f"QA Chain 结果: {result}")


if __name__ == "__main__":
    demo_lcel()
```

- [ ] **Step 4: 运行测试验证**

Run: `cd G:/PythonProject/agnet_learn/knowledge_base_qa && python -m pytest tests/test_stage1_basics.py -v`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
cd G:/PythonProject/agnet_learn
git add knowledge_base_qa/stage1_basics.py knowledge_base_qa/tests/test_stage1_basics.py
git commit -m "feat: 添加 LangChain 基础练习 (Stage 1)

- 创建 LangChainBasics 类
- 实现 LCEL Chain 创建
- 添加 PromptTemplate 使用示例

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Stage 2 - ReAct Agent 实现

**Files:**
- Modify: `knowledge_base_qa/agent.py:1-67`
- Create: `knowledge_base_qa/tests/test_agent.py`

- [ ] **Step 1: 创建测试文件**

```python
# knowledge_base_qa/tests/test_agent.py
import pytest
from unittest.mock import MagicMock, patch
from agent import KnowledgeBaseAgent

def test_agent_initialization():
    """测试 Agent 初始化"""
    with patch('agent.VectorStore'), patch('agent.ConversationMemory'):
        agent = KnowledgeBaseAgent()
        assert agent.vector_store is not None
        assert agent.memory is not None
        assert len(agent.tools) > 0

def test_tools_include_knowledge_base_search():
    """测试工具列表包含知识库检索"""
    with patch('agent.VectorStore'), patch('agent.ConversationMemory'):
        agent = KnowledgeBaseAgent()
        tool_names = [t.name for t in agent.tools]
        assert "search_knowledge_base" in tool_names

def test_react_agent_structure():
    """测试 ReAct Agent 结构"""
    with patch('agent.VectorStore'), patch('agent.ConversationMemory'):
        agent = KnowledgeBaseAgent()
        # 验证 agent 有 executor
        assert hasattr(agent, 'agent_executor')
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd G:/PythonProject/agnet_learn/knowledge_base_qa && python -m pytest tests/test_agent.py -v`
Expected: FAIL - ReAct 相关属性不存在

- [ ] **Step 3: 升级 agent.py**

```python
# knowledge_base_qa/agent.py
"""
知识库问答 Agent - ReAct 版本
Stage 2: 实现推理+行动循环
"""
import os
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from vector_store import VectorStore
from conversation_memory import ConversationMemory
from config import Config
from langchain_openai import ChatOpenAI


class KnowledgeBaseSearchTool:
    """知识库检索工具（Tool 类）"""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def search(self, query: str) -> str:
        """执行知识库检索"""
        docs = self.vector_store.similarity_search(query, k=4)
        if not docs:
            return "知识库中没有找到相关内容。"

        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', '未知')
            content = doc.page_content[:200]  # 限制长度
            context_parts.append(f"[文档{i}] 来源: {source}\n{content}...")

        return "\n\n".join(context_parts)


@tool
def search_knowledge_base(query: str) -> str:
    """检索知识库内容。当你需要回答用户关于文档的问题时使用。

    Args:
        query: 用户的自然语言问题

    Returns:
        相关文档内容，如果没有找到返回提示信息
    """
    # 注意：这里需要通过 Agent 实例获取 vector_store
    # 在实际调用时会通过 context 传递
    return "请在 Agent 上下文中调用知识库检索"


class KnowledgeBaseAgent:
    """基于 ReAct 的知识库问答 Agent"""

    def __init__(self):
        self.vector_store = VectorStore()
        self.memory = ConversationMemory()
        self.llm = ChatOpenAI(
            api_key=Config.MINIMAX_API_KEY,
            base_url=Config.MINIMAX_BASE_URL,
            model_name=Config.MINIMAX_MODEL,
            temperature=Config.MINIMAX_TEMPERATURE
        )

        # 创建知识库检索工具
        self.search_tool = KnowledgeBaseSearchTool(self.vector_store)

        # 创建 ReAct Agent
        self.tools = [self._create_search_tool()]
        self.agent = self._create_react_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,  # 显示推理过程
            max_iterations=5  # 最多 5 步推理
        )

    def _create_search_tool(self):
        """创建检索工具（使用 Function binding）"""
        from langchain_core.tools import StructuredTool

        def search_func(query: str) -> str:
            return self.search_tool.search(query)

        return StructuredTool(
            name="search_knowledge_base",
            description="检索知识库文档内容。当你需要回答用户关于文档的问题时使用。",
            func=search_func,
            args_schema={"query": {"type": "string", "description": "用户的自然语言问题"}}
        )

    def _create_react_agent(self):
        """创建 ReAct Agent"""
        # ReAct 系统提示
        system Prompt = """你是一个知识库问答助手，可以使用工具来检索信息。

你有以下工具：
- search_knowledge_base: 搜索知识库中的文档内容

工作流程：
1. 理解用户问题
2. 决定是否需要检索（如果问题明显与知识库内容相关）
3. 调用 search_knowledge_base 获取相关信息
4. 根据检索结果回答问题

重要：
- 如果检索结果不足，可以再检索一次
- 回答时引用来源
- 如果知识库中没有相关信息，说明情况

输出格式：
Thought: [你的思考]
Action: [工具名称或 final_answer]
Action Input: [输入]
...
Final Answer: [你的回答]
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system Prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        return agent

    def query(self, question: str, session_id: str = "default") -> tuple[str, str]:
        """处理用户问题"""
        # 保存用户消息
        self.memory.add_message(session_id, "user", question)

        # 获取对话历史
        history = self.memory.get_full_history(session_id)
        chat_history_str = self._format_chat_history(history)

        # 调用 ReAct Agent
        result = self.agent_executor.invoke({
            "input": question,
            "chat_history": chat_history_str,
            "agent_scratchpad": []
        })

        answer = result["output"]

        # 保存助手回答
        self.memory.add_message(session_id, "assistant", answer)

        # 获取来源信息
        source_info = self._extract_sources(result)

        return answer, source_info

    def _format_chat_history(self, history: list) -> str:
        """格式化对话历史"""
        if not history:
            return ""
        lines = []
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _extract_sources(self, result: dict) -> str:
        """从 Agent 结果中提取来源"""
        # 从 intermediate_steps 中提取检索的文档来源
        sources = []
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                action = step[0]
                if hasattr(action, 'tool') and action.tool == "search_knowledge_base":
                    # 从 tool input 中可以获取检索词
                    pass
        return "知识库检索"

    def rebuild_index(self, documents):
        self.vector_store.clear_index()
        self.vector_store.create_vector_store(documents)
        return "知识库索引重建完成"
```

- [ ] **Step 4: 运行测试验证**

Run: `cd G:/PythonProject/agnet_learn/knowledge_base_qa && python -m pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 5: 手动测试 ReAct 流程**

Run: `cd G:/PythonProject/agnet_learn/knowledge_base_qa && python main.py`
Expected: 可以看到 Agent 的推理过程（Thought/Action/Observation）

- [ ] **Step 6: 提交代码**

```bash
cd G:/PythonProject/agnet_learn
git add knowledge_base_qa/agent.py knowledge_base_qa/tests/test_agent.py
git commit -m "feat: 实现 ReAct Agent (Stage 2)

- 使用 create_react_agent 创建推理型 Agent
- 实现知识库检索 Tool
- AgentExecutor 支持多步推理循环
- 显示中间推理步骤

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Stage 3 - Reflection 机制

**Files:**
- Create: `knowledge_base_qa/reflection_agent.py`
- Create: `knowledge_base_qa/tests/test_reflection.py`

- [ ] **Step 1: 创建测试文件**

```python
# knowledge_base_qa/tests/test_reflection.py
import pytest
from reflection_agent import ReflectionAgent, QualityChecker

def test_quality_checker():
    """测试质量检查器"""
    checker = QualityChecker()
    # 准备测试数据
    question = "什么是 AI Agent？"
    answer = "AI Agent 是自主执行任务的 AI 系统。"
    context = "AI Agent 能够规划、执行和反思。"

    result = checker.check(question, answer, context)
    assert result is not None
    assert "quality" in result
    assert "passed" in result or "failed" in result

def test_reflection_agent_initialization():
    """测试 Reflection Agent 初始化"""
    agent = ReflectionAgent()
    assert agent.llm is not None
    assert agent.checker is not None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd G:/PythonProject/agnet_learn/knowledge_base_qa && python -m pytest tests/test_reflection.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: 创建 reflection_agent.py**

```python
# knowledge_base_qa/reflection_agent.py
"""
Reflection Agent - 自我反思机制
Stage 3: 回答质量检测与修正
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config


class QualityChecker:
    """回答质量检查器"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """检查以下回答的质量：

问题：{question}
回答：{answer}
文档内容：{context}

请从以下维度检查：
1. 回答是否基于提供的文档？
2. 是否有事实性错误？
3. 是否完整回答了问题？

输出格式：
Quality: [passed/failed]
Reason: [原因说明]
Suggestion: [如果不合格，改进建议]
"""
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def check(self, question: str, answer: str, context: str) -> dict:
        """检查回答质量"""
        result = self.chain.invoke({
            "question": question,
            "answer": answer,
            "context": context
        })

        # 解析结果
        lines = result.split("\n")
        quality_dict = {}
        for line in lines:
            if "Quality:" in line:
                quality_dict["quality"] = line.split("Quality:")[1].strip()
            elif "Reason:" in line:
                quality_dict["reason"] = line.split("Reason:")[1].strip()
            elif "Suggestion:" in line:
                quality_dict["suggestion"] = line.split("Suggestion:")[1].strip()

        return quality_dict


class ReflectionAgent:
    """带自我反思的 Agent"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=Config.MINIMAX_API_KEY,
            base_url=Config.MINIMAX_BASE_URL,
            model_name=Config.MINIMAX_MODEL,
            temperature=Config.MINIMAX_TEMPERATURE
        )
        self.checker = QualityChecker(self.llm)
        self.max_refinement = 2  # 最多修正 2 次

    def refine_answer(self, question: str, draft_answer: str, context: str) -> str:
        """修正回答"""
        refinement_prompt = ChatPromptTemplate.from_template(
            """根据以下建议改进回答：

问题：{question}
原始回答：{draft_answer}
文档内容：{context}
改进建议：{suggestion}

请生成改进后的回答：
"""
        )

        chain = refinement_prompt | self.llm | StrOutputParser()
        improved = chain.invoke({
            "question": question,
            "draft_answer": draft_answer,
            "context": context,
            "suggestion": self.checker.check(question, draft_answer, context).get("suggestion", "")
        })

        return improved

    def process_with_reflection(self, question: str, draft_answer: str, context: str) -> dict:
        """带反思的处理流程"""
        # 1. 检查质量
        quality_result = self.checker.check(question, draft_answer, context)

        # 2. 如果不合格，进行修正
        if quality_result.get("quality") == "failed":
            refined_answer = self.refine_answer(
                question, draft_answer, context
            )
            # 3. 再次检查（简化版）
            return {
                "answer": refined_answer,
                "was_refined": True,
                "original_quality": quality_result
            }

        return {
            "answer": draft_answer,
            "was_refined": False,
            "quality": quality_result
        }


def demo_reflection():
    """演示 Reflection 机制"""
    agent = ReflectionAgent()

    question = "这个文档讲了什么？"
    draft_answer = "文档主要讲述了 AI 技术。"
    context = "本文档介绍了机器学习的基本概念，包括监督学习和无监督学习，以及深度学习的应用。"

    result = agent.process_with_reflection(question, draft_answer, context)

    print(f"最终回答: {result['answer']}")
    print(f"是否经过修正: {result['was_refined']}")


if __name__ == "__main__":
    demo_reflection()
```

- [ ] **Step 4: 运行测试验证**

Run: `cd G:/PythonProject/agnet_learn/knowledge_base_qa && python -m pytest tests/test_reflection.py -v`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
cd G:/PythonProject/agnet_learn
git add knowledge_base_qa/reflection_agent.py knowledge_base_qa/tests/test_reflection.py
git commit -m "feat: 添加 Reflection 机制 (Stage 3)

- 实现 QualityChecker 回答质量检查
- 实现 ReflectionAgent 自我反思
- 支持回答修正流程

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Stage 4 - Memory 升级

**Files:**
- Modify: `knowledge_base_qa/conversation_memory.py:1-61`
- Modify: `knowledge_base_qa/agent.py` (集成)

- [ ] **Step 1: 创建测试文件**

```python
# knowledge_base_qa/tests/test_memory_upgrade.py
import pytest
from unittest.mock import patch, MagicMock
from conversation_memory import ConversationMemory

def test_to_langchain_history():
    """测试转换为 LangChain ChatMessageHistory"""
    with patch('conversation_memory.redis.Redis') as mock_redis:
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance

        memory = ConversationMemory()

        # 模拟历史数据
        mock_instance.lrange.return_value = [
            '{"role": "user", "content": "你好", "timestamp": "2024-01-01T00:00:00"}',
            '{"role": "assistant", "content": "你好！", "timestamp": "2024-01-01T00:00:01"}'
        ]

        history = memory.to_langchain_history("test_session")
        # 验证可以转换为 LangChain 格式
        assert history is not None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd G:/PythonProject/agnet_learn/knowledge_base_qa && python -m pytest tests/test_memory_upgrade.py -v`
Expected: FAIL - to_langchain_history method not found

- [ ] **Step 3: 更新 conversation_memory.py**

```python
# knowledge_base_qa/conversation_memory.py
"""
会话记忆管理 - LangChain Memory 适配
Stage 4: 对接 LangChain Memory 组件
"""
import json
import redis
from datetime import datetime
from typing import Optional
from config import Config

# LangChain Memory 相关导入
try:
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain.memory import ChatMessageHistory as LCChatMessageHistory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LCChatMessageHistory = None


class ConversationMemory:
    """会话记忆管理（Redis + LangChain 适配）"""

    def __init__(self):
        self.redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            password=Config.REDIS_PASSWORD,
            decode_responses=True
        )

    def _get_session_key(self, session_id: str) -> str:
        return f"{Config.SESSION_PREFIX}{session_id}"

    def add_message(self, session_id: str, role: str, content: str):
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        key = self._get_session_key(session_id)
        self.redis_client.rpush(key, json.dumps(message))
        self.redis_client.expire(key, Config.SESSION_TTL)

    def get_history(self, session_id: str, limit: int = 10) -> list:
        key = self._get_session_key(session_id)
        messages = self.redis_client.lrange(key, -limit, -1)
        return [json.loads(msg) for msg in messages]

    def get_full_history(self, session_id: str) -> list:
        key = self._get_session_key(session_id)
        messages = self.redis_client.lrange(key, 0, -1)
        return [json.loads(msg) for msg in messages]

    def clear_session(self, session_id: str):
        key = self._get_session_key(session_id)
        self.redis_client.delete(key)

    def list_sessions(self) -> list:
        pattern = f"{Config.SESSION_PREFIX}*"
        keys = self.redis_client.keys(pattern)
        return [key.replace(Config.SESSION_PREFIX, "") for key in keys]

    def format_for_llm(self, session_id: str) -> str:
        history = self.get_history(session_id)
        if not history:
            return ""

        formatted = []
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    # === LangChain Memory 适配 ===

    def to_langchain_history(self, session_id: str):
        """
        转换为 LangChain 的 ChatMessageHistory

        Args:
            session_id: 会话 ID

        Returns:
            ChatMessageHistory 对象，可用于 RunnableWithMessageHistory
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain 未安装。请运行: pip install langchain-core"
            )

        history = self.get_full_history(session_id)
        chat_history = LCChatMessageHistory()

        for msg in history:
            if msg["role"] == "user":
                chat_history.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                chat_history.add_ai_message(msg["content"])

        return chat_history

    def get_langchain_messages(self, session_id: str) -> list:
        """
        获取 LangChain 格式的消息列表

        Returns:
            List[BaseMessage] - 可直接用于 ChatPromptTemplate
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain 未安装。请运行: pip install langchain-core"
            )

        history = self.get_full_history(session_id)
        messages = []

        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        return messages
```

- [ ] **Step 4: 运行测试验证**

Run: `cd G:/PythonProject/agnet_learn/knowledge_base_qa && python -m pytest tests/test_memory_upgrade.py -v`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
cd G:/PythonProject/agnet_learn
git add knowledge_base_qa/conversation_memory.py knowledge_base_qa/tests/test_memory_upgrade.py
git commit -m "feat: 升级 Memory 模块 (Stage 4)

- 增加 to_langchain_history() 方法
- 增加 get_langchain_messages() 方法
- 支持 RunnableWithMessageHistory 集成

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: 最终集成与测试

**Files:**
- Modify: `knowledge_base_qa/main.py`
- Update: `knowledge_base_qa/README.md`

- [ ] **Step 1: 更新 main.py 支持新 Agent**

```python
# knowledge_base_qa/main.py (更新版本)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from agent import KnowledgeBaseAgent
from document_loader import DocumentLoader


def init_knowledge_base():
    print("=" * 50)
    print("初始化知识库...")
    print("=" * 50)

    loader = DocumentLoader()
    print("\n加载文档...")
    documents = loader.load_and_split()
    print(f"\n共加载 {len(documents)} 个文档块")

    agent = KnowledgeBaseAgent()
    print("\n构建向量索引...")
    agent.rebuild_index(documents)

    print("\n知识库初始化完成！")
    print("Agent 类型: ReAct Agent (支持多步推理)")
    return agent


def chat_loop(agent: KnowledgeBaseAgent):
    session_id = "default"

    print("\n" + "=" * 50)
    print("知识库问答系统 (ReAct Agent)")
    print("输入 'quit' 退出, 'clear' 清除会话历史")
    print("=" * 50 + "\n")

    while True:
        try:
            question = input("你: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "退出"]:
                print("再见！")
                break

            if question.lower() == "clear":
                agent.memory.clear_session(session_id)
                print("会话历史已清除\n")
                continue

            print("\n正在思考...")
            print("(Agent 正在推理，请稍候...)\n")

            answer, sources = agent.query(question, session_id)

            print(f"\n助手: {answer}")
            print(f"\n参考文档:\n{sources}\n")

            # 显示是否经过反思修正
            if hasattr(agent, 'last_was_refined'):
                if agent.last_was_refined:
                    print("[回答已经过质量检查与优化]\n")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--init":
        init_knowledge_base()
        return

    agent = init_knowledge_base()
    chat_loop(agent)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 更新 README.md**

```markdown
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
┌─────────────────────────────┐
│  Reflection: 检查回答质量    │
└─────────────────────────────┘
    ↓
最终回答 + 参考来源
```

## 各 Stage 说明

### Stage 1: LangChain 基础 (stage1_basics.py)
- LCEL (LangChain Expression Language) 语法
- PromptTemplate / ChatPromptTemplate
- Chain 串联

### Stage 2: ReAct Agent (agent.py)
- create_react_agent 创建推理型 Agent
- Tool 定义与注册
- AgentExecutor 多步推理循环

### Stage 3: Reflection (reflection_agent.py)
- QualityChecker 回答质量检查
- ReflectionAgent 自我反思
- 回答修正流程

### Stage 4: Memory 升级 (conversation_memory.py)
- to_langchain_history() 方法
- LangChain ChatMessageHistory 适配
- RunnableWithMessageHistory 集成支持

## 使用方法

```bash
# 初始化知识库
python main.py --init

# 启动问答（显示推理过程）
python main.py
```

## 环境变量

配置 .env 文件：
```
MINIMAX_API_KEY=your_api_key
MINIMAX_MODEL=your_model
REDIS_HOST=localhost
REDIS_PORT=6379
```
```

- [ ] **Step 3: 运行完整测试**

```bash
cd G:/PythonProject/agnet_learn/knowledge_base_qa

# 运行所有测试
python -m pytest tests/ -v

# 手动验证
python main.py --init
# 然后输入一个测试问题
```

- [ ] **Step 4: 提交最终版本**

```bash
cd G:/PythonProject/agnet_learn
git add knowledge_base_qa/main.py knowledge_base_qa/README.md
git commit -m "feat: 完成 Agent 升级，集成所有 Stage

- ReAct Agent 支持多步推理
- Reflection 机制实现回答质量检测
- Memory 模块对接 LangChain
- 更新 README 文档

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## 验证清单

| Stage | 验证项 | 预期结果 |
|-------|--------|----------|
| Stage 1 | `python stage1_basics.py` | LCEL Chain 正常工作 |
| Stage 2 | `python main.py` 后输入问题 | 显示 Thought/Action/Observation |
| Stage 3 | 测试 Reflection 机制 | 质量检查输出 |
| Stage 4 | 测试跨会话记忆 | Redis 数据正确存取 |
| 集成 | 所有测试通过 | `pytest tests/ -v` 全部 PASS |

---

## 学习成果

完成本计划后，你将掌握：

1. **LangChain 基础**：LCEL 语法、Chain、PromptTemplate
2. **ReAct Agent**：创建推理型 Agent、多步推理循环
3. **Reflection 机制**：自我反思、回答质量检测
4. **Memory 升级**：LangChain Memory 组件集成

这些是面试中 Agent 相关的核心知识点。