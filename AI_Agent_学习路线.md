# 吴恩达AI Agent学习路线

## 第一阶段：基础概念（1-2周）

**核心内容：**
- **Agentic AI概念**：理解什么是代理型AI，与传统AI的区别
- **Agent架构**：了解Agent的四大核心组件
  - Planning（规划）
  - Memory（记忆）
  - Tools（工具）
  - Action（行动）

**推荐资源：**
- 吴恩达在Coursera上的"AI Agents"专项课程
- DeepLearning.AI博客文章

---

## 第二阶段：核心技术（2-4周）

### 1. Planning & Reasoning（规划与推理）
- Chain of Thought (CoT)思维链
- Tree of Thoughts (ToT)思维树
- ReAct (Reasoning + Acting)模式

### 2. Tool Use（工具使用）
- Toolformer架构
- 函数调用（Function Calling）
- API集成技术

### 3. Reflection & Self-Correction（反思与自我修正）
- Self-refine机制
- 错误纠正策略
- 迭代优化方法

---

## 第三阶段：高级应用（3-4周）

### 1. Multi-Agent Systems（多智能体系统）
- Agent协作与通信
- 角色分配机制
- 分布式问题解决

### 2. Memory Systems（记忆系统）
- 短期记忆与长期记忆
- RAG (Retrieval-Augmented Generation)集成
- 向量数据库应用

### 3. Practical Projects（实战项目）
- 构建自己的AI Agent
- 自动化工作流设计
- 真实场景应用开发

---

## 第四阶段：工具与框架（持续学习）

### 主流框架
- **LangChain** - Agent开发框架
- **AutoGen** - 微软多Agent框架
- **CrewAI** - 多Agent协作框架
- **LlamaIndex** - 知识检索增强

### 技术栈
```
语言：Python（必须）
向量数据库：Chroma, Pinecone, Weaviate
部署：Docker, Cloud APIs
```

---

## 学习建议

### 学习顺序
1. 先完成Coursera的AI Agents专项课程
2. 复现课程中的示例代码
3. 选择一个框架（推荐LangChain或AutoGen）深入学习
4. 完成2-3个实战项目
5. 关注最新研究论文和社区动态

### 实践建议
- 每天投入2-3小时系统学习
- 注重代码实践，不要只看不练
- 加入相关社区（如Reddit的r/MachineLearning）
- 尝试复现论文中的实验

---

## 推荐学习时间表

| 周次 | 主题 | 目标 |
|------|------|------|
| 1-2 | 基础概念 | 理解Agent架构 |
| 3-4 | Planning & Tools | 掌握ReAct等模式 |
| 5-6 | Reflection机制 | 学会自我修正 |
| 7-8 | Multi-Agent | 理解协作原理 |
| 9-12 | 框架与项目 | 独立开发Agent |
