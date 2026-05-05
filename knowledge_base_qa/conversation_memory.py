import json
import redis
from datetime import datetime
from config import Config


class ConversationMemory:
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
        try:
            from langchain_core.messages import HumanMessage, AIMessage
            from langchain.memory import ChatMessageHistory as LCChatMessageHistory
        except ImportError:
            raise ImportError("LangChain 未安装。请运行: pip install langchain-core")

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
        try:
            from langchain_core.messages import HumanMessage, AIMessage
        except ImportError:
            raise ImportError("LangChain 未安装。请运行: pip install langchain-core")

        history = self.get_full_history(session_id)
        messages = []

        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        return messages
