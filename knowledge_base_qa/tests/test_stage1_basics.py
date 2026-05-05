import pytest
from stage1_basics import create_qa_chain, LangChainBasics

def test_create_qa_chain():
    """测试 LCEL Chain 创建"""
    chain = create_qa_chain()
    assert chain is not None

def test_basics_prompt_template():
    """测试 PromptTemplate 创建"""
    basics = LangChainBasics()
    prompt = basics.create_qa_prompt("什么是 AI Agent")
    # 验证 prompt 包含 question 和 context 变量
    assert "question" in prompt.input_variables
    assert "context" in prompt.input_variables