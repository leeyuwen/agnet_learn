# 吴恩达 Agentic AI 课程学习计划

**课程来源**：https://www.deeplearning.ai/courses/agentic-ai/
**授课教师**：Andrew Ng（吴恩达）
**课程时长**：约5-8小时（自定进度）
**难度级别**：中级
**前置要求**：Python基础，了解LLM基本概念

---

## 课程核心知识点

### 🎯 四大设计模式

#### 1. Reflection（反思模式）
- AI批判自身工作并迭代改进质量
- 类似自动化代码审查
- 实现自我修正机制

#### 2. Tool Use（工具使用模式）
- 连接AI与数据库、API、外部服务
- 实际执行操作，而不仅仅是生成文本
- 扩展AI能力边界

#### 3. Planning（规划模式）
- 将复杂任务分解为可执行步骤
- AI能够遵循并适应计划
- 动态调整执行路径

#### 4. Multi-Agent（多智能体模式）
- 协调多个专业AI系统
- 处理复杂工作流的不同部分
- 分布式任务协作

---

## 📅 学习任务分解（12天学习计划）

### 第一周：基础概念与反思模式

#### 第1-2天：课程概览与环境准备
| 任务 | 内容 | 目标 |
|------|------|------|
| 任务1 | 注册DeepLearning.AI账号 | 获得课程访问权限 |
| 任务2 | 观看课程介绍视频 | 理解Agentic AI核心概念 |
| 任务3 | 配置Python开发环境 | 安装Python、Jupyter Notebook |
| 任务4 | 了解LLM基础概念 | 回顾prompt engineering基础 |

**练习题**：
- 解释传统AI与Agentic AI的区别
- 列举Agentic AI的4个核心设计模式

---

#### 第3-4天：Reflection模式深入学习
| 任务 | 内容 | 目标 |
|------|------|------|
| 任务5 | 观看Reflection模式视频 | 理解自我反思机制原理 |
| 任务6 | 实现基础的Self-Reflection Agent | 能够构建自动改进输出的Agent |
| 任务7 | 完成Reflection代码练习 | 掌握迭代优化方法 |
| 任务8 | 实战：构建代码审查Agent | 应用Reflection模式 |

**代码示例概念**：
```python
# Reflection模式核心逻辑
response = llm.generate(initial_prompt)
for iteration in range(max_iterations):
    critique = llm.critique(response)
    if critique.is_acceptable():
        break
    response = llm.improve(response, critique)
```

**练习题**：
- 为什么Reflection能提高输出质量？
- Reflection模式适合哪些场景？

---

#### 第5-6天：Tool Use模式深入学习
| 任务 | 内容 | 目标 |
|------|------|------|
| 任务9 | 观看Tool Use视频 | 理解工具调用原理 |
| 任务10 | 了解常见工具类型 | 数据库、API、搜索引擎、代码执行 |
| 任务11 | 实现带工具调用的Agent | 掌握function calling技术 |
| 任务12 | 完成Tool Use代码练习 | 连接外部服务 |

**实战项目**：
- 构建能够搜索网页的Research Agent
- 构建能够查询天气的助手Agent

**练习题**：
- Tool Use与普通API调用有什么区别？
- 如何设计可靠的工具错误处理机制？

---

### 第二周：规划模式与多智能体

#### 第7-8天：Planning模式深入学习
| 任务 | 内容 | 目标 |
|------|------|------|
| 任务13 | 观看Planning模式视频 | 理解任务分解与规划 |
| 任务14 | 学习Chain of Thought | 掌握思维链技术 |
| 任务15 | 实现任务规划Agent | 能够分解复杂任务 |
| 任务16 | 学习动态计划调整 | 适应意外情况 |

**核心概念**：
- Task Decomposition（任务分解）
- Dynamic Planning（动态规划）
- Adaptive Execution（自适应执行）

**练习题**：
- Planning模式与简单prompt有什么区别？
- 如何处理计划执行过程中的失败？

---

#### 第9-10天：Multi-Agent模式深入学习
| 任务 | 内容 | 目标 |
|------|------|------|
| 任务17 | 观看Multi-Agent视频 | 理解多智能体协作 |
| 任务18 | 设计多Agent架构 | 角色分配与通信 |
| 任务19 | 实现多Agent协作系统 | 构建Agent团队 |
| 任务20 | 完成Multi-Agent项目 | 综合应用所学知识 |

**实战项目**：
- 构建一个内容创作团队（策划+写作+编辑）
- 构建一个代码开发团队（架构+编码+测试）

**练习题**：
- 多Agent系统相比单Agent有什么优势？
- 如何避免Agent之间的通信冲突？

---

#### 第11-12天：综合实践与评估优化

| 任务 | 内容 | 目标 |
|------|------|------|
| 任务21 | 学习系统评估方法 | 掌握性能指标 |
| 任务22 | 构建测试框架 | 确保Agent可靠性 |
| 任务23 | 错误分析与优化 | 识别并修复常见问题 |
| 任务24 | 生产环境部署 | 了解部署最佳实践 |

**评估要点**：
- 性能指标定义
- 错误分析框架
- 系统优化策略

---

## 📋 学习检查清单

### 基础概念 ✓
- [ ] 理解什么是Agentic AI
- [ ] 掌握4种核心设计模式
- [ ] 了解Agent架构组件

### Reflection模式 ✓
- [ ] 能解释Reflection机制
- [ ] 实现过Self-critique Agent
- [ ] 完成至少1个Reflection练习

### Tool Use模式 ✓
- [ ] 理解工具调用原理
- [ ] 实现过API集成Agent
- [ ] 完成至少1个Tool Use项目

### Planning模式 ✓
- [ ] 掌握任务分解方法
- [ ] 实现过规划Agent
- [ ] 理解动态调整机制

### Multi-Agent模式 ✓
- [ ] 设计过多Agent架构
- [ ] 实现过Agent协作系统
- [ ] 完成至少1个Multi-Agent项目

### 评估与部署 ✓
- [ ] 掌握性能评估方法
- [ ] 构建过测试框架
- [ ] 了解生产环境部署

---

## 🛠️ 实战项目推荐

### 项目1：自动化研究助手
**目标**：构建一个能自动搜集信息、总结报告的Agent
**使用的模式**：Tool Use + Reflection + Planning
**技能点**：网页搜索、信息整合、报告生成

### 项目2：智能代码审查系统
**目标**：构建多Agent代码审查团队
**使用的模式**：Multi-Agent + Reflection
**技能点**：代码分析、问题识别、建议生成

### 项目3：个人AI助手
**目标**：构建能处理日常任务的个人助手
**使用的模式**：全部4种模式
**技能点**：日历管理、邮件处理、信息检索

---

## 📚 延伸学习资源

### 官方文档
- LangChain：https://python.langchain.com/
- LangGraph：https://langchain-ai.github.io/langgraph/
- AutoGen：https://microsoft.github.io/autogen/

### 社区与论文
- Reddit r/MachineLearning
- ReAct论文：https://react-lm.github.io/
- Hugging Face Agents文档

---

## ⚠️ 学习注意事项

1. **注重实践**：每个知识点都要动手编码，而不仅仅是看视频
2. **循序渐进**：不要跳过基础概念直接进入高级主题
3. **记录笔记**：建议使用Jupyter Notebook记录学习心得
4. **完成作业**：课程配套的练习和项目要认真完成
5. **参与讨论**：加入学习社区，与他人交流问题

---

## 📊 学习进度追踪

| 日期 | 已完成任务 | 掌握程度(1-5) | 备注 |
|------|-----------|--------------|------|
| Day 1 | | | |
| Day 2 | | | |
| Day 3 | | | |
| Day 4 | | | |
| ... | | | |

**自我评估标准**：
- 1分：听说过概念
- 2分：理解原理
- 3分：能完成基础代码
- 4分：能独立完成项目
- 5分：能教学他人
