# 04 - 高级Agent开发

## 项目目标

掌握高级Agent设计模式、Multi-Agent协作、自定义工具开发和Agent优化技巧，能够设计和实现复杂的Agent系统。

## 核心设计理念

基于 **aelx 的 Agent Skills 设计哲学**：

### 设计层次
```
❌ 下限：碰运气完成任务
   → 只在理想条件下能跑通

✅ 及格线：能观察、能诊断、能避免卡死
   → 有日志、状态检查、错误处理

🌟 状元：能闭环、能迭代、直到验证通过
   → 测试-修复-验证循环

🚀 高阶：能降低认知负荷
   → 清晰接口、脚手架、高信息密度
```

## 学习内容

### 核心模块

#### 1. Agent设计模式
- ✅ **ReAct模式**：推理(Reasoning) + 行动(Acting)
- ✅ **Plan-and-Execute**：先规划后执行
- ✅ **Reflection**：自我反思和改进
- ✅ **Multi-Agent协作**：多个Agent协同工作

#### 2. 工具开发
- ✅ 自定义Tool的编写
- ✅ Tool的错误处理
- ✅ Tool的权限控制
- ✅ 常用工具集成

#### 3. Agent优化
- ✅ Prompt优化技巧
- ✅ 减少幻觉的策略
- ✅ 成本控制和缓存
- ✅ 并发和异步处理
- ✅ 错误处理和重试机制

#### 4. Agent设计最佳实践（核心）
- ✅ 降低认知负荷的设计
- ✅ 闭环验证设计
- ✅ 避免常见陷阱
- ✅ 使用子Agent实现异步

## 文件说明

### 配置文件
- `CLAUDE.md` - Claude Code工作指南
- `README.md` - 本文件
- `快速开始指南.md` - 5分钟快速入门
- `requirements.txt` - 依赖包列表
- `.env.example` - 环境变量模板

### 示例代码（8个）

#### 基础模式（3个）
1. **`01_react_pattern.py`** - ReAct模式实现
   - Thought → Action → Observation循环
   - 工具调用和推理结合
   - 详细日志和验证

2. **`02_plan_execute.py`** - Plan-and-Execute模式
   - 任务分解和规划
   - 按计划执行
   - 动态调整计划

3. **`03_reflection_agent.py`** - Reflection模式
   - 自我评估和改进
   - 从失败中学习
   - 经验积累

#### Multi-Agent（2个）
4. **`04_multi_agent_basic.py`** - Multi-Agent基础
   - 多个Agent协作
   - 任务分配
   - 结果汇总

5. **`05_multi_agent_advanced.py`** - Multi-Agent高级
   - 复杂协作模式
   - 冲突解决
   - 迭代改进

#### 工具和优化（2个）
6. **`06_tool_development.py`** - 工具开发
   - 自定义工具编写
   - 工具验证和测试
   - 错误处理

7. **`07_agent_optimization.py`** - Agent优化
   - Token优化
   - 并发处理
   - 缓存策略

#### 综合项目（1个）
8. **`08_research_assistant.py`** - 研究助手系统
   - 完整的Multi-Agent系统
   - 研究员 + 编辑 + 审核员
   - 闭环验证和迭代改进

### 自定义工具（tools/）
- `web_scraper.py` - 网页抓取工具
- `code_executor.py` - 代码执行工具
- `file_manager.py` - 文件管理工具
- `api_caller.py` - API调用工具

### Agent实现（agents/）
- `researcher.py` - 研究员Agent
- `editor.py` - 编辑Agent
- `reviewer.py` - 审核员Agent
- `coordinator.py` - 协调员Agent

### 测试文件（tests/）
- `test_react_pattern.py` - ReAct模式测试
- `test_multi_agent.py` - Multi-Agent测试
- `test_tools.py` - 工具测试

### 文档
- `学习笔记.md` - 学习记录和心得
- `实践项目指南.md` - 项目实践指导
- `FAQ.md` - 常见问题解答

## 快速开始

### 1. 安装依赖
```bash
cd "04_高级Agent开发"
pip install -r requirements.txt
```

### 2. 配置API密钥
```bash
# Windows
copy .env.example .env
notepad .env

# Linux/Mac
cp .env.example .env
nano .env
```

填入你的API密钥：
```
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. 运行示例

#### 按顺序学习（推荐）
```bash
# 1. ReAct模式
python 01_react_pattern.py

# 2. Plan-and-Execute
python 02_plan_execute.py

# 3. Reflection
python 03_reflection_agent.py

# 4. Multi-Agent基础
python 04_multi_agent_basic.py

# 5. Multi-Agent高级
python 05_multi_agent_advanced.py

# 6. 工具开发
python 06_tool_development.py

# 7. Agent优化
python 07_agent_optimization.py

# 8. 研究助手项目
python 08_research_assistant.py
```

#### 运行测试
```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_react_pattern.py -v
```

## 核心概念

### Agent设计模式对比

| 模式 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **ReAct** | 需要工具调用的任务 | 灵活、可解释 | 可能陷入循环 |
| **Plan-and-Execute** | 复杂多步骤任务 | 结构清晰 | 计划可能不准确 |
| **Reflection** | 需要改进的任务 | 能自我优化 | 需要多次迭代 |
| **Multi-Agent** | 需要专业分工的任务 | 专业化、可扩展 | 协调复杂 |

### ReAct模式流程

```
开始
  ↓
Thought: 我需要做什么？
  ↓
Action: 调用工具
  ↓
Observation: 观察结果
  ↓
Thought: 结果如何？需要继续吗？
  ↓
[继续循环 或 返回最终答案]
```

### Multi-Agent协作模式

```
任务
  ↓
协调员Agent
  ├─→ 研究员Agent → 收集信息
  ├─→ 编辑Agent → 整理内容
  └─→ 审核员Agent → 验证质量
  ↓
汇总结果
  ↓
[通过 或 迭代改进]
```

## 实践项目

### 必做项目（3个）

#### 1. 研究助手Agent（基础）
**目标**：构建一个能够搜索、总结、保存的Agent

**要求**：
- ✅ 使用ReAct模式
- ✅ 集成搜索工具
- ✅ 实现闭环验证
- ✅ 添加详细日志

**验证标准**：
- 能成功搜索并总结信息
- 有完整的日志记录
- 能验证搜索结果质量
- 能处理搜索失败的情况

#### 2. Multi-Agent研究系统（进阶）
**目标**：构建研究员+编辑+审核员的协作系统

**要求**：
- ✅ 实现3个专业Agent
- ✅ 实现协调机制
- ✅ 实现迭代改进循环
- ✅ 添加质量评估

**验证标准**：
- 研究员能收集相关信息
- 编辑能整理成结构化内容
- 审核员能提出改进建议
- 系统能迭代改进直到通过

#### 3. 代码审查Agent（高级）
**目标**：构建能够审查代码质量的Agent系统

**要求**：
- ✅ 分析代码结构
- ✅ 检查常见问题
- ✅ 提供改进建议
- ✅ 生成审查报告

**验证标准**：
- 能识别代码问题
- 建议具体可行
- 报告结构清晰
- 能处理不同语言

### 可选项目（3个）

1. **数据分析Agent**
   - 读取数据文件
   - 自动分析和可视化
   - 生成分析报告

2. **内容生成系统**
   - 多Agent协作生成内容
   - 自动优化和改进
   - 质量评估

3. **自动化测试Agent**
   - 生成测试用例
   - 执行测试
   - 分析测试结果

## 学习路径

### Week 7（第一周）

**Day 1-2：Agent设计模式**
- [ ] 学习ReAct模式
- [ ] 运行`01_react_pattern.py`
- [ ] 理解Thought-Action-Observation循环
- [ ] 实现一个简单的ReAct Agent

**Day 3-4：Plan-and-Execute & Reflection**
- [ ] 学习Plan-and-Execute模式
- [ ] 运行`02_plan_execute.py`
- [ ] 学习Reflection模式
- [ ] 运行`03_reflection_agent.py`

**Day 5-7：Multi-Agent基础**
- [ ] 学习Multi-Agent协作
- [ ] 运行`04_multi_agent_basic.py`
- [ ] 实现简单的Multi-Agent系统
- [ ] 完成研究助手项目（基础版）

### Week 8（第二周）

**Day 1-2：Multi-Agent高级**
- [ ] 学习复杂协作模式
- [ ] 运行`05_multi_agent_advanced.py`
- [ ] 实现迭代改进机制

**Day 3-4：工具开发和优化**
- [ ] 学习工具开发
- [ ] 运行`06_tool_development.py`
- [ ] 学习Agent优化
- [ ] 运行`07_agent_optimization.py`

**Day 5-7：综合项目**
- [ ] 运行`08_research_assistant.py`
- [ ] 完成Multi-Agent研究系统
- [ ] 总结学习心得
- [ ] 准备进入下一阶段

## 设计检查清单

在实现Agent时，使用 [Agent_设计检查清单.md](../Agent_设计检查清单.md)：

### 设计前
- [ ] 这个任务真的需要Agent吗？
- [ ] 任务的复杂度如何？
- [ ] 如何验证任务完成？

### 实现时
- [ ] 能查看日志吗？
- [ ] 能检查状态吗？
- [ ] 能验证输出吗？
- [ ] 有错误处理吗？
- [ ] 避免阻塞命令了吗？

### 完成后
- [ ] 有自动化测试吗？
- [ ] 实现了闭环验证吗？
- [ ] 文档完善吗？
- [ ] 代码质量如何？

## 常见问题

### 1. Agent陷入循环怎么办？
**解决方案**：
- 设置最大迭代次数
- 检测重复模式
- 提供退出条件

### 2. 工具调用失败怎么办？
**解决方案**：
- 提供清晰的工具描述
- 实现参数验证
- 添加降级策略

### 3. Multi-Agent如何协调？
**解决方案**：
- 使用协调员Agent
- 定义清晰的接口
- 实现消息传递机制

### 4. 如何优化Token使用？
**解决方案**：
- 总结历史对话
- 使用缓存
- 压缩prompt

### 5. 如何验证Agent质量？
**解决方案**：
- 编写自动化测试
- 使用评估指标
- 人工审查

## 学习资源

### 必读文档
- [Agent_设计最佳实践.md](../Agent_设计最佳实践.md) ⭐ 必读
- [Agent_设计检查清单.md](../Agent_设计检查清单.md) ⭐ 实践参考
- [AI_Agent_Learning_Roadmap.md](../AI_Agent_Learning_Roadmap.md)

### 推荐阅读
- [ReAct论文](https://arxiv.org/abs/2210.03629)
- [LangChain Agent文档](https://python.langchain.com/docs/modules/agents/)
- [AutoGPT项目](https://github.com/Significant-Gravitas/AutoGPT)

### 视频教程
- DeepLearning.AI - Building Systems with ChatGPT
- LangChain官方教程

## 技能检查

完成本阶段后，你应该能够：

### 基础技能
- [ ] 理解不同Agent设计模式
- [ ] 实现ReAct模式Agent
- [ ] 开发自定义工具
- [ ] 处理工具调用错误

### 中级技能
- [ ] 实现Plan-and-Execute模式
- [ ] 实现Reflection机制
- [ ] 构建Multi-Agent系统
- [ ] 优化Token使用

### 高级技能
- [ ] 设计复杂的Agent架构
- [ ] 实现闭环验证
- [ ] 降低Agent认知负荷
- [ ] 处理各种边界情况

## 评估标准

### 代码质量（40分）
- [ ] 代码结构清晰（10分）
- [ ] 有完整的文档字符串（10分）
- [ ] 有错误处理（10分）
- [ ] 有日志记录（10分）

### 功能完整性（30分）
- [ ] 实现了所有要求的功能（15分）
- [ ] 功能正常工作（15分）

### 验证和测试（20分）
- [ ] 有自动化测试（10分）
- [ ] 有闭环验证（10分）

### 设计质量（10分）
- [ ] 遵循设计最佳实践（5分）
- [ ] 降低了认知负荷（5分）

**总分**：100分
**及格线**：60分
**优秀**：80分以上

## 下一步

完成本阶段后，你将进入：

**第五阶段：模型部署与调优**
- 本地模型部署
- 模型量化
- LoRA微调
- 推理优化

---

**创建日期**：2026-02-06
**预计完成**：2周（Week 7-8）

记住：**验证驱动、闭环思维、降低认知负荷** 🚀
