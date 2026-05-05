import pytest
from unittest.mock import patch, MagicMock
import sys

# Mock langchain modules before importing ConversationMemory
mock_human_message = MagicMock()
mock_ai_message = MagicMock()
mock_chat_message_history = MagicMock()

mock_langchain_core = MagicMock()
mock_langchain_core.messages = MagicMock()
mock_langchain_core.messages.HumanMessage = mock_human_message
mock_langchain_core.messages.AIMessage = mock_ai_message

mock_langchain = MagicMock()
mock_langchain.memory = MagicMock()
mock_langchain.memory.ChatMessageHistory = mock_chat_message_history

# Create a mock for the entire langchain hierarchy
mock_langchain_module = MagicMock()
mock_langchain_module.langchain_core = mock_langchain_core
mock_langchain_module.langchain = mock_langchain

sys.modules['langchain'] = mock_langchain
sys.modules['langchain.memory'] = mock_langchain.memory
sys.modules['langchain_core'] = mock_langchain_core
sys.modules['langchain_core.messages'] = mock_langchain_core.messages

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
        assert history is not None
        mock_chat_message_history.assert_called_once()
