# 02 - Agent框架入门

## 项目目标
学习使用LangChain等主流Agent框架，理解Agent的核心组件和工作原理，掌握不同复杂度任务的解决方案选择。

## 学习内容
- ✅ LangChain核心组件：Models, Prompts, Chains
- ✅ LangChain Agents和Tools
- ✅ Memory机制
- ✅ 其他Agent框架对比
- ✅ 任务场景分类和框架选择

## 文件说明

### 1. `requirements.txt`
项目依赖包列表

### 2. `.env.example`
环境变量模板文件

### 3. `01_langchain_basics.py`
LangChain基础组件示例：Models, Prompts, Chains

### 4. `02_langchain_agents.py`
LangChain Agent和Tools使用示例

### 5. `03_memory_demo.py`
Memory机制演示：对话历史管理

### 6. `04_agent_with_tools.py`
带工具调用的完整Agent示例（计算器、搜索、天气）

### 7. `05_workflow_vs_agent.py`
Workflow和Agent的对比示例

### 8. `06_framework_comparison.py`
不同框架的对比演示

### 9. `tools/`
自定义工具目录
- `calculator.py` - 计算器工具
- `weather.py` - 天气查询工具
- `search.py` - 搜索工具

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置API密钥
复制 `.env.example` 为 `.env`，并填入你的API密钥：
```bash
cp .env.example .env
```

### 3. 运行示例
```bash
# LangChain基础
python 01_langchain_basics.py

# Agent和Tools
python 02_langchain_agents.py

# Memory演示
python 03_memory_demo.py

# 完整Agent示例
python 04_agent_with_tools.py

# Workflow vs Agent对比
python 05_workflow_vs_agent.py

# 框架对比
python 06_framework_comparison.py
```

## 核心概念

### Agent vs Workflow
```
简单查询 → 直接API调用
  ├─ 单次问答
  └─ 简单信息提取

中等复杂度 → Workflow/Chain
  ├─ 多步骤处理
  ├─ 条件分支
  └─ 数据转换流程

高复杂度 → Agent系统
  ├─ 需要工具调用
  ├─ 动态决策
  ├─ 多轮交互
  └─ 自主规划
```

### LangChain核心组件

#### 1. Models
- LLM包装器
- 统一的接口
- 支持多种模型提供商

#### 2. Prompts
- PromptTemplate
- ChatPromptTemplate
- Few-shot examples

#### 3. Chains
- LLMChain：基础链
- SequentialChain：顺序执行
- RouterChain：条件路由

#### 4. Agents
- ReAct Agent：推理+行动
- OpenAI Functions Agent
- Structured Chat Agent

#### 5. Tools
- 工具定义和注册
- 工具调用和错误处理
- 自定义工具开发

#### 6. Memory
- ConversationBufferMemory
- ConversationSummaryMemory
- VectorStoreMemory

## 框架对比

### LangChain
- **优点**：生态完善、组件丰富、社区活跃
- **缺点**：抽象层次多、学习曲线陡
- **适用**：复杂Agent系统、需要多种工具集成

### LlamaIndex
- **优点**：专注数据索引、RAG性能好
- **缺点**：功能相对单一
- **适用**：文档问答、知识库检索

### AutoGPT
- **优点**：自主性强、任务分解好
- **缺点**：成本高、不稳定
- **适用**：研究和实验、自主任务执行

### CrewAI
- **优点**：多Agent协作、角色定义清晰
- **缺点**：相对新、生态较小
- **适用**：多角色协作场景

## 最佳实践

### 1. 选择合适的抽象层次
- 简单任务不要过度设计
- 复杂任务才需要Agent
- 优先考虑Workflow

### 2. 工具设计原则
- 单一职责
- 清晰的输入输出
- 完善的错误处理
- 详细的描述文档

### 3. Memory管理
- 根据场景选择Memory类型
- 注意token限制
- 定期清理历史

### 4. 成本控制
- 使用缓存减少重复调用
- 选择合适的模型
- 设置max_iterations限制

### 5. 调试技巧
- 启用verbose模式
- 使用callbacks记录日志
- 单独测试每个工具

## 实践项目

### 项目1：带工具的研究助手
构建一个能够搜索、总结、保存的研究助手Agent。

**功能**：
- 搜索相关资料
- 总结关键信息
- 保存到文件

### 项目2：文档处理Workflow
实现一个文档处理流程：上传→解析→摘要→翻译→保存。

**要求**：
- 使用SequentialChain
- 处理不同格式文档
- 错误处理和重试

### 项目3：多工具Agent
创建一个集成多种工具的通用Agent。

**工具包括**：
- 计算器
- 天气查询
- 网络搜索
- 文件操作

## 下一步
完成本章节后，进入 `03_向量数据库与RAG` 学习知识库构建。

---
**创建日期**：2026-01-25
**状态**：进行中
