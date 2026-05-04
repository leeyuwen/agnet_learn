import os
from langchain_community.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage
from vector_store import VectorStore
from conversation_memory import ConversationMemory
from config import Config
from langchain_openai import ChatOpenAI

class KnowledgeBaseAgent:
    def __init__(self):
        self.vector_store = VectorStore()
        self.memory = ConversationMemory()
        self.llm = ChatOpenAI(
            api_key=Config.MINIMAX_API_KEY,
            base_url=Config.MINIMAX_BASE_URL,
            model_name=Config.MINIMAX_MODEL,
            temperature=Config.MINIMAX_TEMPERATURE
        )
        self.tools = [
            Tool(
                name="知识库检索",
                func=self.vector_store.similarity_search,
                description="当用户询问关于文档内容的问题时使用此工具。输入应该是用户的自然语言问题。"
            )
        ]

    def query(self, question: str, session_id: str = "default") -> str:
        self.memory.add_message(session_id, "user", question)

        history = self.memory.format_for_llm(session_id)

        if history:
            system_msg = f"""你是一个知识库问答助手。基于提供的文档内容回答用户的问题。

对话历史:
{history}"""
        else:
            system_msg = "你是一个知识库问答助手。基于提供的文档内容回答用户的问题。"

        docs = self.vector_store.similarity_search(question, k=4)
        context = "\n".join([doc.page_content for doc in docs])
        source_info = "\n".join([f"- {doc.metadata.get('source', '未知')}" for doc in docs])

        full_prompt = f"""基于以下文档内容回答问题。

文档内容:
{context}

问题: {question}

请根据文档内容回答，如果文档中没有相关信息，请说明。"""

        response = self.llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=full_prompt)
        ])

        answer = response.content
        self.memory.add_message(session_id, "assistant", answer)

        return answer, source_info

    def rebuild_index(self, documents):
        self.vector_store.clear_index()
        self.vector_store.create_vector_store(documents)
        return "知识库索引重建完成"
