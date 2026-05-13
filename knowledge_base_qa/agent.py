"""
知识库问答 Agent - ReAct 版本
Stage 2: 实现推理+行动循环
"""
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from vector_store import VectorStore
from conversation_memory import ConversationMemory
from config import Config
from langchain_openai import ChatOpenAI


class KnowledgeBaseSearchTool:
    """知识库检索工具"""

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
            content = doc.page_content[:200]
            context_parts.append(f"[文档{i}] 来源: {source}\n{content}...")

        return "\n\n".join(context_parts)


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

        self.search_tool = KnowledgeBaseSearchTool(self.vector_store)

        # 创建 ReAct Agent
        self.tools = [self._create_search_tool()]
        self.agent = self._create_react_agent()

    def _create_search_tool(self):
        """创建检索工具"""
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
        system_prompt = """你是一个知识库问答助手，可以使用工具来检索信息。

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

        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )
        return agent

    def query(self, question: str, session_id: str = "default") -> tuple[str, str]:
        """处理用户问题"""
        self.memory.add_message(session_id, "user", question)

        history = self.memory.get_full_history(session_id)

        # 构建消息列表
        messages = []
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        result = self.agent.invoke(
            {"messages": messages},
            config=RunnableConfig(max_iterations=5, verbose=True)
        )

        # 从结果中提取助手回复
        all_messages = result.get("messages", [])
        answer = "无法生成回答"

        for msg in reversed(all_messages):
            if hasattr(msg, 'content') and msg.content and hasattr(msg, 'type') and msg.type == 'ai':
                answer = msg.content
                break

        self.memory.add_message(session_id, "assistant", answer)

        source_info = self._extract_sources(result)

        return answer, source_info

    def _format_chat_history(self, history: list) -> str:
        """格式化对话历史"""
        if not history:
            return ""
        lines = []
        for msg in history:
            lines.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(lines)

    def _extract_sources(self, result: dict) -> str:
        """从 Agent 结果中提取来源"""
        return "知识库检索"

    def rebuild_index(self, documents):
        self.vector_store.clear_index()
        self.vector_store.create_vector_store(documents)
        return "知识库索引重建完成"