"""
Multi-Agent System - 多智能体协作系统
Stage 5: 实现 Supervisor + Workers 协作模式

核心架构：
- SupervisorAgent: 协调者，分析任务并分配给合适的 Worker
- ResearcherAgent: 研究员，负责检索知识库
- WriterAgent: 写作者，负责整理和生成最终回答
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from config import Config
from vector_store import VectorStore
from conversation_memory import ConversationMemory


# ============================================================
# 基础工具定义
# ============================================================

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


# ============================================================
# Worker Agents
# ============================================================

class ResearcherAgent:
    """研究员 Agent - 负责检索知识库"""

    def __init__(self):
        self.vector_store = None  # 懒加载
        self.llm = ChatOpenAI(
            api_key=Config.MINIMAX_API_KEY,
            base_url=Config.MINIMAX_BASE_URL,
            model_name=Config.MINIMAX_MODEL,
            temperature=Config.MINIMAX_TEMPERATURE
        )
        self.search_tool = None
        self.agent = None

    def _lazy_init(self):
        """懒加载资源"""
        if self.vector_store is None:
            self.vector_store = VectorStore()
            self.search_tool = self._create_search_tool()
            self.agent = self._create_agent()

    def _create_search_tool(self):
        """创建检索工具"""
        vector_store = self.vector_store

        def search_func(query: str) -> str:
            docs = vector_store.similarity_search(query, k=4)
            if not docs:
                return "知识库中没有找到相关内容。"

            context_parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', '未知')
                content = doc.page_content[:200]
                context_parts.append(f"[文档{i}] 来源: {source}\n{content}...")

            return "\n\n".join(context_parts)

        return StructuredTool(
            name="search_knowledge_base",
            description="检索知识库文档内容",
            func=search_func,
            args_schema={"query": {"type": "string", "description": "用户的自然语言问题"}}
        )

    def _create_agent(self):
        """创建研究员 Agent"""
        system_prompt = """你是一个专业的研究员，专注于从知识库中检索相关信息。

你的职责：
1. 分析用户问题，确定检索关键词
2. 使用 search_knowledge_base 工具检索相关文档
3. 对检索结果进行筛选，提取最相关的内容
4. 返回结构化的检索结果

输出格式：
- 检索使用的关键词
- 找到的相关文档列表
- 每个文档的核心内容摘要
"""
        return create_agent(
            model=self.llm,
            tools=[self.search_tool],
            system_prompt=system_prompt
        )

    def research(self, query: str) -> dict:
        """执行研究任务"""
        self._lazy_init()

        result = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=RunnableConfig(max_iterations=3, verbose=False)
        )

        messages = result.get("messages", [])
        research_output = ""

        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                research_output = msg.content
                break

        return {
            "query": query,
            "research_result": research_output,
            "source": "knowledge_base"
        }


class WriterAgent:
    """写作者 Agent - 负责整理和生成最终回答"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=Config.MINIMAX_API_KEY,
            base_url=Config.MINIMAX_BASE_URL,
            model_name=Config.MINIMAX_MODEL,
            temperature=Config.MINIMAX_TEMPERATURE
        )

    def write(self, question: str, research_data: dict) -> str:
        """根据研究数据生成最终回答"""
        prompt = ChatPromptTemplate.from_template("""
根据以下研究数据回答用户问题。

用户问题：{question}

研究数据：
{research_data}

请生成一个完整、准确、有条理的回答。
- 如果研究数据中有相关信息，基于这些信息回答
- 如果没有相关信息，说明情况
- 适当引用来源
""")
        chain = prompt | self.llm | StrOutputParser()

        result = chain.invoke({
            "question": question,
            "research_data": research_data.get("research_result", "无相关信息")
        })

        return result


# ============================================================
# Supervisor Agent
# ============================================================

class SupervisorAgent:
    """监督者 Agent - 负责任务协调和分发"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=Config.MINIMAX_API_KEY,
            base_url=Config.MINIMAX_BASE_URL,
            model_name=Config.MINIMAX_MODEL,
            temperature=Config.MINIMAX_TEMPERATURE
        )

        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()

    def analyze_task(self, question: str) -> dict:
        """分析任务，决定需要哪些 Worker 参与"""
        analysis_prompt = ChatPromptTemplate.from_template("""
分析以下用户问题，确定回答所需的步骤。

用户问题：{question}

请确定：
1. 是否需要检索知识库？（回答是/否）
2. 是否需要整理和生成回答？（回答是/否）
3. 任务的复杂程度（简单/中等/复杂）

输出格式：
NeedResearch: [是/否]
NeedWrite: [是/否]
Complexity: [简单/中等/复杂]
""")
        chain = analysis_prompt | self.llm | StrOutputParser()

        result = chain.invoke({"question": question})

        decision = {"need_research": False, "need_write": True, "complexity": "简单"}

        for line in result.split("\n"):
            if "NeedResearch:" in line:
                decision["need_research"] = "是" in line
            elif "NeedWrite:" in line:
                decision["need_write"] = "是" in line
            elif "Complexity:" in line:
                decision["complexity"] = line.split("Complexity:")[1].strip()

        return decision

    def process(self, question: str) -> dict:
        """协调处理用户问题"""
        # 1. 分析任务
        decision = self.analyze_task(question)

        # 2. 根据决策执行任务
        research_data = {}

        if decision["need_research"]:
            research_data = self.researcher.research(question)

        # 3. 生成最终回答
        if decision["need_write"] and research_data:
            final_answer = self.writer.write(question, research_data)
        elif research_data:
            final_answer = research_data.get("research_result", "无法生成回答")
        else:
            final_answer = "无法回答这个问题"

        return {
            "question": question,
            "answer": final_answer,
            "research_data": research_data,
            "decision": decision
        }


# ============================================================
# Multi-Agent System 整合
# ============================================================

class MultiAgentSystem:
    """多智能体系统"""

    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.memory = ConversationMemory()

    def query(self, question: str, session_id: str = "default") -> str:
        """处理用户问题"""
        self.memory.add_message(session_id, "user", question)

        result = self.supervisor.process(question)
        answer = result["answer"]

        self.memory.add_message(session_id, "assistant", answer)

        return answer


# ============================================================
# Demo
# ============================================================

def demo_multi_agent():
    """演示多智能体协作"""
    print("=" * 60)
    print("Multi-Agent System 演示")
    print("=" * 60)

    system = MultiAgentSystem()

    test_questions = [
        "LangChain 是什么？",
        "解释 LCEL 的工作原理",
    ]

    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 40)

        result = system.supervisor.process(question)

        print(f"决策: {result['decision']}")
        print(f"回答: {result['answer'][:200]}...")

    print("\n" + "=" * 60)


def demo_supervisor_only():
    """仅演示 Supervisor 分析功能"""
    print("=" * 60)
    print("Supervisor 分析演示")
    print("=" * 60)

    supervisor = SupervisorAgent()

    test_questions = [
        "LangChain 是什么？",
        "你好",
        "帮我写一首诗",
    ]

    for question in test_questions:
        print(f"\n问题: {question}")
        decision = supervisor.analyze_task(question)
        print(f"决策: {decision}")


if __name__ == "__main__":
    demo_supervisor_only()
    print("\n")
    demo_multi_agent()