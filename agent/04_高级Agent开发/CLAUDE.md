# CLAUDE.md - 04_高级Agent开发

This file provides guidance to Claude Code when working with code in this directory.

## 项目概述

第四阶段：高级Agent开发 - 学习Agent设计模式、Multi-Agent协作、工具开发和Agent优化技巧。

## 核心设计原则（基于 aelx 的设计哲学）

### 验证驱动开发
- 每个Agent功能都要有验证机制
- 不要只是"写完就完"，要"测试-修复-验证"
- 使用日志、状态检查、输出验证

### 闭环思维
- 实现 → 测试 → 发现问题 → 修复 → 再测试
- 从"凭直觉修"进化到"用测试定位问题"
- API产出：用curl测试
- 网页产出：用Playwright截图

### 降低认知负荷
- 设计清晰的工具接口和文档
- 提供脚手架和护栏
- 提高信息密度，避免让Agent猜测

## 开发命令

### Python 环境

```bash
# 激活虚拟环境（如果使用）
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 运行示例

```bash
# 按顺序运行示例
python 01_react_pattern.py          # ReAct模式
python 02_plan_execute.py           # Plan-and-Execute模式
python 03_reflection_agent.py       # Reflection模式
python 04_multi_agent_basic.py      # Multi-Agent基础
python 05_multi_agent_advanced.py   # Multi-Agent高级
python 06_tool_development.py       # 工具开发
python 07_agent_optimization.py     # Agent优化
python 08_research_assistant.py     # 研究助手项目

# 运行测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_react_pattern.py -v
```

## 架构概览

```
04_高级Agent开发/
├── CLAUDE.md                    # 本文件
├── README.md                    # 项目说明
├── 快速开始指南.md              # 快速入门
├── requirements.txt             # 依赖包
├── .env.example                 # 环境变量模板
│
├── 01_react_pattern.py          # ReAct模式示例
├── 02_plan_execute.py           # Plan-and-Execute模式
├── 03_reflection_agent.py       # Reflection模式
├── 04_multi_agent_basic.py      # Multi-Agent基础
├── 05_multi_agent_advanced.py   # Multi-Agent高级
├── 06_tool_development.py       # 工具开发
├── 07_agent_optimization.py     # Agent优化
├── 08_research_assistant.py     # 研究助手项目
│
├── tools/                       # 自定义工具
│   ├── __init__.py
│   ├── web_scraper.py          # 网页抓取工具
│   ├── code_executor.py        # 代码执行工具
│   ├── file_manager.py         # 文件管理工具
│   └── api_caller.py           # API调用工具
│
├── agents/                      # Agent实现
│   ├── __init__.py
│   ├── researcher.py           # 研究员Agent
│   ├── editor.py               # 编辑Agent
│   ├── reviewer.py             # 审核员Agent
│   └── coordinator.py          # 协调员Agent
│
├── examples/                    # 完整示例
│   ├── research_system/        # 研究系统
│   └── code_review_system/     # 代码审查系统
│
├── tests/                       # 测试文件
│   ├── test_react_pattern.py
│   ├── test_multi_agent.py
│   └── test_tools.py
│
└── 学习笔记.md                  # 学习记录
```

## 关键约定

### Agent设计层次

**下限（避免）**：
```python
# ❌ 只管调用，不管验证
def deploy():
    result = agent.run("部署应用")
    return "完成"  # 真的完成了吗？
```

**及格线（最低要求）**：
```python
# ✅ 有日志、状态检查、验证
def deploy():
    logger.info("开始部署...")
    result = agent.run("部署应用")

    # 验证
    if not check_deployment_status():
        raise Exception("部署失败")

    logger.info(f"部署成功: {result}")
    return result
```

**状元（目标）**：
```python
# 🌟 闭环验证，自动修复
def deploy_with_verification():
    for attempt in range(3):
        try:
            # 部署
            result = deploy()

            # 测试
            test_results = run_tests(result.url)

            # 验证通过
            if test_results.all_passed:
                return result

            # 失败，诊断并修复
            issue = diagnose(test_results)
            fix(issue)

        except Exception as e:
            logger.error(f"尝试 {attempt + 1} 失败: {e}")

    raise Exception("部署失败")
```

### 工具开发规范

**必须包含**：
- 清晰的文档字符串（说明用途、参数、返回值、注意事项）
- 参数验证
- 错误处理
- 日志记录
- 使用示例

**示例**：
```python
def search_web(query: str, max_results: int = 10) -> List[SearchResult]:
    """
    使用Google搜索网页

    Args:
        query: 搜索关键词（1-100个字符）
        max_results: 最多返回结果数（1-100）

    Returns:
        List[SearchResult]: 搜索结果列表

    Raises:
        ValueError: query为空或过长
        RateLimitError: 超过速率限制

    示例:
        # 好的调用
        results = search_web("Python tutorial")

        # 坏的调用
        results = search_web("")  # ValueError

    注意事项:
        - 需要设置GOOGLE_API_KEY环境变量
        - 速率限制：每分钟10次
        - 不要在循环中频繁调用
    """
    # 参数验证
    if not query or len(query) > 100:
        raise ValueError("query必须在1-100个字符之间")

    # 实现...
```

### Multi-Agent协作模式

**基础模式**：
```python
# 顺序执行
researcher = ResearcherAgent()
editor = EditorAgent()
reviewer = ReviewerAgent()

# 研究 → 编辑 → 审核
research_result = researcher.run(task)
edited_result = editor.run(research_result)
final_result = reviewer.run(edited_result)
```

**高级模式（闭环）**：
```python
# 迭代改进
for iteration in range(max_iterations):
    # 研究
    research = researcher.run(task)

    # 编辑
    draft = editor.run(research)

    # 审核
    review = reviewer.run(draft)

    # 通过？
    if review.approved:
        return draft

    # 不通过，根据反馈改进
    task = improve_task(task, review.feedback)
```

### 验证和测试

**每个Agent都要有测试**：
```python
def test_researcher_agent():
    agent = ResearcherAgent()

    # 测试基本功能
    result = agent.run("Python最佳实践")
    assert result is not None
    assert len(result.sources) > 0

    # 测试错误处理
    with pytest.raises(ValueError):
        agent.run("")  # 空查询

    # 测试边界情况
    result = agent.run("a" * 1000)  # 超长查询
    assert result is not None
```

**集成测试**：
```python
def test_multi_agent_system():
    system = ResearchSystem()

    # 运行完整流程
    result = system.run("AI Agent最佳实践")

    # 验证结果
    assert result.research is not None
    assert result.draft is not None
    assert result.review.approved

    # 验证质量
    assert len(result.draft) > 1000
    assert result.review.score >= 8.0
```

## 常见问题和解决方案

### Agent陷入循环

**问题**：Agent重复相同的动作

**解决方案**：
```python
# 设置最大迭代次数
agent = Agent(max_iterations=10)

# 检测重复模式
action_history = []
for action in agent.run(task):
    if action in action_history[-3:]:  # 最近3次重复
        raise Exception("检测到循环，终止执行")
    action_history.append(action)
```

### 工具调用失败

**问题**：工具参数错误或工具不可用

**解决方案**：
```python
# 提供清晰的工具描述
@tool
def search_web(query: str) -> str:
    """
    搜索网页

    参数:
        query: 搜索关键词（必需，1-100字符）

    返回:
        搜索结果的JSON字符串

    错误:
        - ValueError: query为空或过长
        - RateLimitError: 超过速率限制
    """
    pass

# 实现降级策略
try:
    result = search_web(query)
except RateLimitError:
    # 降级：使用缓存结果
    result = get_cached_results(query)
```

### 幻觉问题

**问题**：Agent编造不存在的工具或结果

**解决方案**：
```python
# 强制引用工具输出
prompt = """
规则：
1. 只能使用提供的工具
2. 必须引用工具的实际输出
3. 不要编造结果

示例：
✅ 正确：根据search_web()返回的结果，...
❌ 错误：我搜索了网页，发现...（没有引用实际输出）
"""

# 使用结构化输出
from pydantic import BaseModel

class AgentResponse(BaseModel):
    thought: str
    action: str
    action_input: dict
    observation: str  # 必须来自工具输出
```

## 性能优化

### Token使用优化

```python
# ❌ 每次都发送完整历史
agent.run(task, history=full_history)

# ✅ 总结历史，只保留关键信息
summarized_history = summarize(full_history)
agent.run(task, history=summarized_history)
```

### 并发处理

```python
# ❌ 顺序执行
results = []
for task in tasks:
    results.append(agent.run(task))

# ✅ 并发执行
import asyncio

async def run_tasks(tasks):
    return await asyncio.gather(*[
        agent.arun(task) for task in tasks
    ])

results = asyncio.run(run_tasks(tasks))
```

### 缓存结果

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def search_web(query: str):
    # 相同查询直接返回缓存
    return _search_web_impl(query)
```

## 调试技巧

### 启用详细日志

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# LangChain verbose模式
agent = Agent(verbose=True)
```

### 使用Callbacks

```python
from langchain.callbacks import StdOutCallbackHandler

agent = Agent(
    callbacks=[StdOutCallbackHandler()],
    verbose=True
)
```

### 单步调试

```python
# 手动执行每一步
for step in agent.plan(task):
    print(f"执行: {step}")
    result = agent.execute_step(step)
    print(f"结果: {result}")

    # 检查结果
    if not validate(result):
        print("验证失败，停止执行")
        break
```

## 最佳实践

### 1. 设计前先思考

使用 [Agent_设计检查清单.md](../Agent_设计检查清单.md)：
- [ ] 这个任务真的需要Agent吗？
- [ ] 如何验证任务完成？
- [ ] 哪些命令会阻塞？
- [ ] 如何降低认知负荷？

### 2. 实现时遵循层次

- 先达到及格线（有日志、验证、错误处理）
- 再追求状元（闭环验证、自动修复）
- 最后优化到高阶（降低认知负荷）

### 3. 测试驱动开发

- 先写测试，再写实现
- 每个功能都要有测试
- 测试要覆盖正常、错误、边界情况

### 4. 持续改进

- 记录遇到的问题和解决方案
- 总结可复用的模式
- 更新文档和示例

## 参考文档

- [Agent_设计最佳实践.md](../Agent_设计最佳实践.md) - 必读
- [Agent_设计检查清单.md](../Agent_设计检查清单.md) - 实践时参考
- [AI_Agent_Learning_Roadmap.md](../AI_Agent_Learning_Roadmap.md) - 完整路线

## Git提交规范

```
<类型>: <中文简短描述>

## 变更内容
- 变更1
- 变更2

## 修改文件
- file1.py: 说明
- file2.py: 说明

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

类型: feat | fix | refactor | docs | test | chore

---

**创建日期**: 2026-02-06
**阶段**: 第四阶段 - 高级Agent开发

记住: **验证驱动、闭环思维、降低认知负荷** 🚀
