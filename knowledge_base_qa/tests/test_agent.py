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
        # 验证 agent 存在
        assert hasattr(agent, 'agent')