import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    return agent


def chat_loop(agent: KnowledgeBaseAgent):
    session_id = "default"

    print("\n" + "=" * 50)
    print("知识库问答系统")
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
            answer, sources = agent.query(question, session_id)

            print(f"\n助手: {answer}")
            print(f"\n参考文档:\n{sources}\n")

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
