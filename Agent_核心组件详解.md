# Agentic AI 核心组件详解：Planning 与 Memory

## 什么是 Agentic AI？

| 类型 | 流程 | 特点 |
|------|------|------|
| **传统AI** | 输入 → 输出（一次性完成） | 被动响应 |
| **Agentic AI** | 输入 → 思考 → 行动 → 观察 → 迭代（自主循环） | 主动规划 |

---

## 1. Planning（规划）

### 什么是 Planning？

Planning 是 Agent 将复杂任务分解为可执行步骤的能力。

### 核心实现模式

#### 1.1 思维链（Chain of Thought, CoT）

```python
# 传统方式：直接给出答案
user: 买苹果的花费是葡萄的2倍，葡萄3元，苹果多少钱？
model: 6元  # 直接回答，无过程

# CoT方式：展示推理过程
user: 买苹果的花费是葡萄的2倍，葡萄3元，苹果多少钱？
model:
  - 葡萄的价格是3元
  - 苹果的价格是葡萄的2倍
  - 所以苹果 = 3 × 2 = 6元
  - 答案是6元
```

#### 1.2 思维树（Tree of Thoughts, ToT）

```python
# 对于复杂问题，探索多条路径

问题：如何让公司收入增长？
├── 方案A：开拓新市场
│   ├── 进入东南亚市场 → 风险：本地化成本高
│   └── 进入农村市场 → 风险：基础设施不完善
├── 方案B：产品差异化
│   ├── 高端化 → 风险：用户群体缩小
│   └── 降低成本 → 风险：质量下降
└── 方案C：并购竞争对手
    └── 风险：整合困难
```

#### 1.3 ReAct 模式（Reasoning + Acting）

```python
# 结合推理和行动，持续交互
agent_loop:
  thought: "用户想要查询北京的天气"
  action: "调用天气API[北京]"
  observation: "API返回：晴，25度"
  thought: "现在我有了天气信息，可以回答用户了"
  output: "北京今天天气晴朗，气温25度"
```

### Planning 代码示例

```python
class PlanningAgent:
    def plan(self, task):
        # 1. 任务拆解
        steps = self.decompose(task)

        # 2. 排序优先级
        ordered_steps = self.prioritize(steps)

        # 3. 规划执行路径
        execution_plan = self.create_execution_plan(ordered_steps)

        return execution_plan

    def decompose(self, task):
        """将任务分解为子任务"""
        prompt = f"""
        将这个任务分解成具体步骤：
        任务：{task}

        输出格式：
        1. 步骤1
        2. 步骤2
        ...
        """
        return llm.invoke(prompt).split('\n')
```

---

## 2. Memory（记忆）

### 什么是 Memory？

Memory 是 Agent 存储和检索历史信息的能力，模拟人类的记忆系统。

### 记忆的类型

| 类型 | 特性 | 存储位置 | 生命周期 |
|------|------|----------|----------|
| **感官记忆** | 原始输入 | 缓存 | 毫秒-秒 |
| **短期记忆** | 当前上下文 | LLM context window | 对话期间 |
| **长期记忆** | 持久知识 | 向量数据库 | 持久 |

### 短期记忆实现

```python
class ShortTermMemory:
    def __init__(self, max_turns=10):
        self.conversation_history = []
        self.max_turns = max_turns

    def add(self, role, content):
        """添加对话记录"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })

        # 限制长度，防止超出 context window
        if len(self.conversation_history) > self.max_turns:
            self.conversation_history.pop(0)

    def get_context(self):
        """获取完整的上下文"""
        return "\n".join([
            f"{m['role']}: {m['content']}"
            for m in self.conversation_history
        ])
```

### 长期记忆实现（向量数据库）

```python
class LongTermMemory:
    def __init__(self):
        self.vector_store = Chroma()  # 向量数据库

    def add_memory(self, content, metadata=None):
        """存储记忆"""
        embedding = embed(content)
        self.vector_store.add(
            vectors=[embedding],
            documents=[content],
            metadatas=[metadata or {}]
        )

    def retrieve(self, query, top_k=5):
        """基于语义检索记忆"""
        query_embedding = embed(query)
        results = self.vector_store.search(
            query_vectors=[query_embedding],
            n_results=top_k
        )
        return results['documents']

    def think(self, current_context):
        """思考：结合历史记忆"""
        relevant_memories = self.retrieve(current_context)
        return f"根据你的记忆：{relevant_memories}"
```

### 完整的记忆系统架构

```python
class AgentMemory:
    def __init__(self):
        self.short_term = ShortTermMemory(max_turns=10)
        self.long_term = LongTermMemory()

    def remember(self, query):
        """检索相关记忆"""
        # 1. 先查短期记忆
        recent = self.short_term.get_context()

        # 2. 再查长期记忆（语义相似度）
        relevant_long_term = self.long_term.retrieve(query)

        return {
            "recent_conversation": recent,
            "relevant_experiences": relevant_long_term
        }

    def learn(self, experience):
        """从经验中学习，存入长期记忆"""
        self.long_term.add_memory(
            content=experience,
            metadata={"timestamp": datetime.now()}
        )
```

---

## 3. 两者结合：完整 Agent 循环

```python
class Agent:
    def __init__(self):
        self.memory = AgentMemory()
        self.planner = PlanningAgent()
        self.tools = ToolSet()

    def run(self, task):
        # 1. 记忆：获取相关上下文
        context = self.memory.remember(task)

        # 2. 规划：分解任务
        plan = self.planner.plan(task, context)

        # 3. 执行循环
        for step in plan:
            # 3a. 行动
            result = self.execute(step)

            # 3b. 观察
            observation = self.tools.observe(result)

            # 3c. 反思（存入记忆）
            self.memory.learn(observation)

            # 3d. 调整计划（如需要）
            plan = self.planner.adjust(plan, observation)

        return final_result
```

---

## 总结

| 组件 | 核心问题 | 解决方案 |
|------|----------|----------|
| **Planning** | 如何分解和执行复杂任务？ | CoT、ToT、ReAct |
| **Memory** | 如何存储和检索信息？ | 短期记忆（上下文）+ 长期记忆（向量检索）|

---

## 学习资源

- 吴恩达 Coursera "AI Agents" 专项课程
- DeepLearning.AI 博客文章
- ReAct 论文：[Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- CoT 论文：[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)