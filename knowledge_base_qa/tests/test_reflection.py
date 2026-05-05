import pytest
from reflection_agent import ReflectionAgent, QualityChecker

def test_quality_checker():
    """测试质量检查器"""
    checker = QualityChecker()
    question = "什么是 AI Agent？"
    answer = "AI Agent 是自主执行任务的 AI 系统。"
    context = "AI Agent 能够规划、执行和反思。"

    result = checker.check(question, answer, context)
    assert result is not None
    assert "quality" in result
    assert "passed" in result["quality"] or "failed" in result["quality"]

def test_reflection_agent_initialization():
    """测试 Reflection Agent 初始化"""
    agent = ReflectionAgent()
    assert agent.llm is not None
    assert agent.checker is not None