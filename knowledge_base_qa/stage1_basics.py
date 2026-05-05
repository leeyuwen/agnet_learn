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

请根据上下文回答问题。
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