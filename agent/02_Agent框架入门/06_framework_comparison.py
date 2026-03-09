"""
不同Agent框架对比演示
对比LangChain、LlamaIndex等框架的特点
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain import hub

# 加载环境变量
load_dotenv()


# ============ LangChain示例 ============

def demo_langchain():
    """演示LangChain框架"""
    print("=" * 60)
    print("1. LangChain框架演示")
    print("=" * 60)
    print("\n特点: 组件丰富、生态完善、社区活跃")
    print("-" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 定义工具
    def search_tool(query: str) -> str:
        """模拟搜索"""
        results = {
            "langchain": "LangChain是一个用于开发LLM应用的框架，提供了丰富的组件。",
            "python": "Python是一种高级编程语言，广泛用于AI和数据科学。",
            "ai": "人工智能是计算机科学的一个分支，致力于创建智能机器。"
        }
        for key, value in results.items():
            if key in query.lower():
                return value
        return "未找到相关信息"

    def calculator_tool(expr: str) -> str:
        """计算器"""
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return str(result)
        except:
            return "计算错误"

    tools = [
        Tool(
            name="Search",
            func=search_tool,
            description="搜索信息。输入搜索关键词。"
        ),
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="计算数学表达式。"
        )
    ]

    # 创建Agent
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # 测试
    test_query = "搜索LangChain的信息，然后计算2+2"
    print(f"\n问题: {test_query}")
    print("-" * 60)

    result = agent_executor.invoke({"input": test_query})
    print(f"\n结果: {result['output']}")

    print("\n✅ LangChain优势:")
    print("  • 组件丰富（Chains, Agents, Memory等）")
    print("  • 支持多种LLM提供商")
    print("  • 活跃的社区和文档")
    print("  • 灵活的抽象层次")

    print("\n⚠️  LangChain劣势:")
    print("  • 学习曲线较陡")
    print("  • 抽象层次多，有时过于复杂")
    print("  • API变化较快")


# ============ LlamaIndex示例（概念演示）============

def demo_llamaindex_concept():
    """演示LlamaIndex的概念（不需要实际安装）"""
    print("\n" + "=" * 60)
    print("2. LlamaIndex框架概念")
    print("=" * 60)
    print("\n特点: 专注数据索引和检索，RAG性能优秀")
    print("-" * 60)

    concept_code = '''
# LlamaIndex典型用法示例（概念代码）

from llama_index import VectorStoreIndex, SimpleDirectoryReader

# 1. 加载文档
documents = SimpleDirectoryReader('data').load_data()

# 2. 创建索引
index = VectorStoreIndex.from_documents(documents)

# 3. 查询
query_engine = index.as_query_engine()
response = query_engine.query("你的问题")

print(response)
'''

    print("\n代码示例:")
    print(concept_code)

    print("\n✅ LlamaIndex优势:")
    print("  • 专注于数据索引和检索")
    print("  • RAG系统性能优秀")
    print("  • 简单易用的API")
    print("  • 支持多种数据源")
    print("  • 优秀的文档分块策略")

    print("\n⚠️  LlamaIndex劣势:")
    print("  • 功能相对单一（主要是RAG）")
    print("  • Agent功能不如LangChain丰富")
    print("  • 生态相对较小")

    print("\n💡 适用场景:")
    print("  → 文档问答系统")
    print("  → 知识库检索")
    print("  → RAG应用")
    print("  → 需要高质量检索的场景")


# ============ AutoGPT概念 ============

def demo_autogpt_concept():
    """演示AutoGPT的概念"""
    print("\n" + "=" * 60)
    print("3. AutoGPT框架概念")
    print("=" * 60)
    print("\n特点: 自主任务执行，强调Agent的自主性")
    print("-" * 60)

    concept = '''
AutoGPT工作流程:

1. 接收目标
   用户: "研究并写一篇关于AI的文章"

2. 任务分解
   Agent自动分解为:
   - 搜索AI相关资料
   - 阅读和总结信息
   - 组织文章结构
   - 撰写文章内容
   - 审核和修改

3. 自主执行
   Agent自动执行每个步骤，无需人工干预

4. 自我反思
   Agent评估结果，决定是否需要改进

5. 迭代优化
   根据反思结果，重新执行或优化
'''

    print(concept)

    print("\n✅ AutoGPT优势:")
    print("  • 高度自主，减少人工干预")
    print("  • 能处理复杂的长期任务")
    print("  • 自我反思和改进能力")
    print("  • 任务分解能力强")

    print("\n⚠️  AutoGPT劣势:")
    print("  • 成本高（大量LLM调用）")
    print("  • 可能陷入循环")
    print("  • 结果不稳定")
    print("  • 难以控制和调试")

    print("\n💡 适用场景:")
    print("  → 研究和探索任务")
    print("  → 需要高度自主性的场景")
    print("  → 实验和原型开发")
    print("  → 不敏感成本的应用")


# ============ CrewAI概念 ============

def demo_crewai_concept():
    """演示CrewAI的概念"""
    print("\n" + "=" * 60)
    print("4. CrewAI框架概念")
    print("=" * 60)
    print("\n特点: 多Agent协作，角色分工明确")
    print("-" * 60)

    concept_code = '''
# CrewAI典型用法示例（概念代码）

from crewai import Agent, Task, Crew

# 定义Agent角色
researcher = Agent(
    role='研究员',
    goal='收集和分析信息',
    backstory='你是一个经验丰富的研究员...'
)

writer = Agent(
    role='作家',
    goal='撰写高质量文章',
    backstory='你是一个专业的技术作家...'
)

editor = Agent(
    role='编辑',
    goal='审核和改进内容',
    backstory='你是一个严谨的编辑...'
)

# 定义任务
research_task = Task(
    description='研究AI Agent的最新进展',
    agent=researcher
)

write_task = Task(
    description='根据研究结果撰写文章',
    agent=writer
)

edit_task = Task(
    description='审核并改进文章',
    agent=editor
)

# 创建团队
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task]
)

# 执行
result = crew.kickoff()
'''

    print("\n代码示例:")
    print(concept_code)

    print("\n✅ CrewAI优势:")
    print("  • 多Agent协作机制完善")
    print("  • 角色定义清晰")
    print("  • 适合团队协作场景")
    print("  • 任务分配灵活")

    print("\n⚠️  CrewAI劣势:")
    print("  • 相对较新，生态较小")
    print("  • 文档和示例较少")
    print("  • 成本较高（多个Agent）")

    print("\n💡 适用场景:")
    print("  → 需要多角色协作的任务")
    print("  → 复杂的内容生成")
    print("  → 模拟团队工作流程")
    print("  → 需要专业分工的场景")


# ============ 框架对比总结 ============

def framework_comparison_table():
    """框架对比表格"""
    print("\n" + "=" * 60)
    print("5. 框架对比总结")
    print("=" * 60)

    comparison = """
╔══════════════════════════════════════════════════════════════╗
║                    Agent框架对比表                           ║
╚══════════════════════════════════════════════════════════════╝

┌────────────┬──────────┬──────────┬──────────┬──────────┐
│   特性     │LangChain │LlamaIndex│ AutoGPT  │ CrewAI   │
├────────────┼──────────┼──────────┼──────────┼──────────┤
│ 学习曲线   │   中等   │   简单   │   复杂   │   中等   │
│ 组件丰富度 │   ★★★★★ │   ★★★   │   ★★★   │   ★★★   │
│ RAG能力    │   ★★★★  │   ★★★★★ │   ★★    │   ★★★   │
│ Agent能力  │   ★★★★★ │   ★★★   │   ★★★★★ │   ★★★★  │
│ 多Agent    │   ★★★   │   ★★    │   ★★    │   ★★★★★ │
│ 成本控制   │   ★★★★  │   ★★★★  │   ★★    │   ★★★   │
│ 稳定性     │   ★★★★  │   ★★★★★ │   ★★    │   ★★★   │
│ 社区活跃度 │   ★★★★★ │   ★★★★  │   ★★★   │   ★★★   │
│ 文档质量   │   ★★★★  │   ★★★★  │   ★★★   │   ★★★   │
│ 生产就绪   │   ★★★★  │   ★★★★★ │   ★★    │   ★★★   │
└────────────┴──────────┴──────────┴──────────┴──────────┘

╔══════════════════════════════════════════════════════════════╗
║                    使用场景推荐                              ║
╚══════════════════════════════════════════════════════════════╝

🎯 LangChain - 通用Agent开发
   ✓ 需要丰富的组件和工具
   ✓ 需要灵活的抽象层次
   ✓ 复杂的Agent系统
   ✓ 需要社区支持

📚 LlamaIndex - 文档问答和RAG
   ✓ 文档问答系统
   ✓ 知识库检索
   ✓ 需要高质量检索
   ✓ RAG应用

🤖 AutoGPT - 自主任务执行
   ✓ 研究和探索
   ✓ 需要高度自主性
   ✓ 实验性项目
   ✓ 不敏感成本

👥 CrewAI - 多Agent协作
   ✓ 需要角色分工
   ✓ 团队协作模拟
   ✓ 复杂内容生成
   ✓ 多步骤工作流

╔══════════════════════════════════════════════════════════════╗
║                    选择决策树                                ║
╚══════════════════════════════════════════════════════════════╝

                    开始
                     │
                     ▼
              主要需求是什么？
                     │
        ┌────────────┼────────────┬────────────┐
        │            │            │            │
        ▼            ▼            ▼            ▼
    文档检索    通用Agent    自主执行    多角色协作
        │            │            │            │
        ▼            ▼            ▼            ▼
   LlamaIndex   LangChain    AutoGPT      CrewAI

╔══════════════════════════════════════════════════════════════╗
║                    实践建议                                  ║
╚══════════════════════════════════════════════════════════════╝

1. 新手入门
   → 从LangChain开始，学习基础概念
   → 然后根据需求选择专门框架

2. 生产环境
   → LangChain或LlamaIndex（稳定性好）
   → 避免使用AutoGPT（成本和稳定性问题）

3. 原型开发
   → 快速验证想法：LlamaIndex（RAG）或LangChain（Agent）
   → 实验性功能：AutoGPT

4. 成本考虑
   → 成本敏感：LangChain + 精心设计的Workflow
   → 成本不敏感：AutoGPT或CrewAI

5. 混合使用
   → 可以组合使用多个框架
   → 例如：LangChain + LlamaIndex（Agent + RAG）

6. 未来趋势
   → 框架会越来越融合
   → 关注标准化接口（如LangChain Expression Language）
   → 学习核心概念比学习特定框架更重要
"""

    print(comparison)


# ============ 实际选择建议 ============

def practical_recommendations():
    """实际项目选择建议"""
    print("\n" + "=" * 60)
    print("6. 实际项目选择建议")
    print("=" * 60)

    recommendations = """
📋 项目类型 → 推荐框架

1. 客服聊天机器人
   推荐: LangChain
   理由: 需要Memory、工具调用、灵活的对话管理

2. 文档问答系统
   推荐: LlamaIndex
   理由: 专注RAG，检索质量高，API简单

3. 代码助手
   推荐: LangChain
   理由: 需要多种工具（搜索、执行、测试）

4. 研究助手
   推荐: LangChain或CrewAI
   理由: 需要多步骤推理和工具调用

5. 内容生成系统
   推荐: CrewAI
   理由: 多角色协作（研究、写作、编辑）

6. 数据分析Agent
   推荐: LangChain
   理由: 需要灵活的工具集成和数据处理

7. 个人知识库
   推荐: LlamaIndex
   理由: 专注检索，易于维护

8. 自动化任务执行
   推荐: AutoGPT（实验）或LangChain（生产）
   理由: 根据稳定性需求选择

💡 通用建议:

• 80%的场景用LangChain就够了
• 需要RAG时考虑LlamaIndex
• 多Agent协作考虑CrewAI
• AutoGPT适合实验，不适合生产

• 先用简单方案，不要过度设计
• 可以从一个框架开始，后续迁移
• 关注核心概念，而非特定框架
"""

    print(recommendations)


def main():
    """主函数"""
    print("\n🔍 Agent框架对比演示\n")

    try:
        # 1. LangChain演示
        demo_langchain()

        # 2. LlamaIndex概念
        demo_llamaindex_concept()

        # 3. AutoGPT概念
        demo_autogpt_concept()

        # 4. CrewAI概念
        demo_crewai_concept()

        # 5. 框架对比
        framework_comparison_table()

        # 6. 实际建议
        practical_recommendations()

        print("\n" + "=" * 60)
        print("✅ 所有演示完成！")
        print("=" * 60)

        print("\n💡 核心要点:")
        print("1. 不同框架有不同的优势和适用场景")
        print("2. LangChain是最通用的选择")
        print("3. LlamaIndex专注RAG，性能优秀")
        print("4. AutoGPT适合实验，不适合生产")
        print("5. CrewAI适合多Agent协作场景")
        print("6. 根据项目需求选择合适的框架")
        print("7. 可以混合使用多个框架")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请检查API配置和网络连接")


if __name__ == "__main__":
    main()
