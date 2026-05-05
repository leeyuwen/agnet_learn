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

    def __init__(self, llm: ChatOpenAI = None):
        if llm is None:
            llm = ChatOpenAI(
                api_key=Config.MINIMAX_API_KEY,
                base_url=Config.MINIMAX_BASE_URL,
                model_name=Config.MINIMAX_MODEL,
                temperature=Config.MINIMAX_TEMPERATURE
            )
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
        quality_result = self.checker.check(question, draft_answer, context)

        if quality_result.get("quality") == "failed":
            refined_answer = self.refine_answer(
                question, draft_answer, context
            )
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