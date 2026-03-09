"""
Workflow vs Agent 对比演示
展示何时使用Workflow，何时使用Agent
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain import hub
import time

# 加载环境变量
load_dotenv()


# ============ Workflow示例 ============

def demo_workflow_fixed_steps():
    """演示固定步骤的Workflow"""
    print("=" * 60)
    print("1. Workflow示例 - 固定步骤流程")
    print("=" * 60)
    print("\n场景: 文章处理流程（翻译 → 摘要 → 评分）")
    print("-" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 步骤1: 翻译
    translate_prompt = PromptTemplate(
        input_variables=["text"],
        template="将以下中文翻译成英文:\n\n{text}\n\n英文翻译:"
    )
    translate_chain = LLMChain(
        llm=llm,
        prompt=translate_prompt,
        output_key="translation"
    )

    # 步骤2: 摘要
    summary_prompt = PromptTemplate(
        input_variables=["translation"],
        template="用一句话总结以下英文内容:\n\n{translation}\n\n摘要:"
    )
    summary_chain = LLMChain(
        llm=llm,
        prompt=summary_prompt,
        output_key="summary"
    )

    # 步骤3: 评分
    score_prompt = PromptTemplate(
        input_variables=["summary"],
        template="对以下摘要的质量打分(1-10分)，并简要说明理由:\n\n{summary}\n\n评分:"
    )
    score_chain = LLMChain(
        llm=llm,
        prompt=score_prompt,
        output_key="score"
    )

    # 组合成顺序链
    workflow = SequentialChain(
        chains=[translate_chain, summary_chain, score_chain],
        input_variables=["text"],
        output_variables=["translation", "summary", "score"],
        verbose=True
    )

    # 测试
    input_text = """
    人工智能正在改变我们的生活。从智能手机到自动驾驶汽车，
    AI技术无处不在。它帮助我们更高效地工作，更好地理解世界。
    """

    print(f"\n输入文本:\n{input_text.strip()}")
    print("\n" + "=" * 60)

    start_time = time.time()
    result = workflow.invoke({"text": input_text})
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("结果:")
    print("=" * 60)
    print(f"\n1. 翻译:\n{result['translation']}")
    print(f"\n2. 摘要:\n{result['summary']}")
    print(f"\n3. 评分:\n{result['score']}")
    print(f"\n⏱️  耗时: {elapsed:.2f}秒")


def demo_workflow_conditional():
    """演示带条件分支的Workflow"""
    print("\n" + "=" * 60)
    print("2. Workflow示例 - 条件分支")
    print("=" * 60)
    print("\n场景: 内容审核流程（检测 → 分类 → 处理）")
    print("-" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 步骤1: 内容分类
    classify_prompt = PromptTemplate(
        input_variables=["content"],
        template="""分析以下内容的类型，只回答一个词: 技术、生活、娱乐、其他

内容: {content}

类型:"""
    )
    classify_chain = LLMChain(
        llm=llm,
        prompt=classify_prompt,
        output_key="category"
    )

    # 步骤2: 生成标签
    tag_prompt = PromptTemplate(
        input_variables=["content", "category"],
        template="""为以下{category}类内容生成3个标签（用逗号分隔）:

{content}

标签:"""
    )
    tag_chain = LLMChain(
        llm=llm,
        prompt=tag_prompt,
        output_key="tags"
    )

    # 组合
    workflow = SequentialChain(
        chains=[classify_chain, tag_chain],
        input_variables=["content"],
        output_variables=["category", "tags"],
        verbose=True
    )

    # 测试
    test_contents = [
        "Python是一种流行的编程语言，广泛用于数据科学和AI开发。",
        "今天天气真好，适合出去散步。",
        "最新的电影上映了，评分很高。"
    ]

    for i, content in enumerate(test_contents, 1):
        print(f"\n测试 {i}:")
        print(f"内容: {content}")
        print("-" * 60)

        result = workflow.invoke({"content": content})
        print(f"分类: {result['category']}")
        print(f"标签: {result['tags']}")


# ============ Agent示例 ============

def demo_agent_dynamic_tools():
    """演示Agent的动态工具选择"""
    print("\n" + "=" * 60)
    print("3. Agent示例 - 动态工具选择")
    print("=" * 60)
    print("\n场景: 智能助手（根据问题自动选择工具）")
    print("-" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 定义工具
    def calculator(expr: str) -> str:
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return str(result)
        except:
            return "计算错误"

    def text_length(text: str) -> str:
        return f"长度: {len(text)} 字符"

    def reverse_text(text: str) -> str:
        return text[::-1]

    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="计算数学表达式"
        ),
        Tool(
            name="TextLength",
            func=text_length,
            description="获取文本长度"
        ),
        Tool(
            name="ReverseText",
            func=reverse_text,
            description="反转文本"
        )
    ]

    # 创建Agent
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )

    # 测试不同类型的问题
    questions = [
        "计算 100 * 5 + 20",
        "'Hello World' 有多长？",
        "把 'Python' 反转",
        "计算 50 + 30，然后告诉我结果有几个字符"  # 需要多个工具
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 60)

        start_time = time.time()
        result = agent_executor.invoke({"input": question})
        elapsed = time.time() - start_time

        print(f"\n答案: {result['output']}")
        print(f"⏱️  耗时: {elapsed:.2f}秒")


# ============ 对比分析 ============

def comparison_analysis():
    """对比分析Workflow和Agent"""
    print("\n" + "=" * 60)
    print("4. Workflow vs Agent 对比分析")
    print("=" * 60)

    comparison = """
╔════════════════════════════════════════════════════════════╗
║                    Workflow vs Agent                       ║
╚════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────┐
│ WORKFLOW (Chain)                                            │
├─────────────────────────────────────────────────────────────┤
│ 特点:                                                       │
│   • 固定的执行步骤                                          │
│   • 预定义的流程                                            │
│   • 确定性输出                                              │
│   • 可预测的成本                                            │
│                                                             │
│ 优点:                                                       │
│   ✓ 执行效率高                                              │
│   ✓ 成本可控                                                │
│   ✓ 易于调试                                                │
│   ✓ 结果稳定                                                │
│                                                             │
│ 缺点:                                                       │
│   ✗ 缺乏灵活性                                              │
│   ✗ 无法处理意外情况                                        │
│   ✗ 不能动态调整                                            │
│                                                             │
│ 适用场景:                                                   │
│   → 固定的数据处理流程                                      │
│   → ETL任务                                                 │
│   → 文档转换（翻译、摘要、格式化）                          │
│   → 批量处理                                                │
│   → 需要严格控制流程的场景                                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ AGENT                                                       │
├─────────────────────────────────────────────────────────────┤
│ 特点:                                                       │
│   • 动态决策                                                │
│   • 自主选择工具                                            │
│   • 迭代式推理                                              │
│   • 适应性强                                                │
│                                                             │
│ 优点:                                                       │
│   ✓ 高度灵活                                                │
│   ✓ 能处理复杂任务                                          │
│   ✓ 自主规划                                                │
│   ✓ 适应不同场景                                            │
│                                                             │
│ 缺点:                                                       │
│   ✗ 成本较高（多次LLM调用）                                 │
│   ✗ 可能不稳定                                              │
│   ✗ 调试困难                                                │
│   ✗ 可能陷入循环                                            │
│                                                             │
│ 适用场景:                                                   │
│   → 需要工具调用的任务                                      │
│   → 问答系统                                                │
│   → 研究助手                                                │
│   → 需要动态决策的场景                                      │
│   → 用户意图不明确的情况                                    │
└─────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════╗
║                      决策树                                ║
╚════════════════════════════════════════════════════════════╝

                    开始
                     │
                     ▼
            步骤是否固定？
                 ╱   ╲
               是      否
              ╱          ╲
             ▼            ▼
        需要工具？      Agent
          ╱   ╲
        否      是
       ╱          ╲
      ▼            ▼
  Workflow      考虑成本
                 ╱   ╲
            成本敏感   不敏感
              ╱          ╲
             ▼            ▼
        Workflow       Agent

╔════════════════════════════════════════════════════════════╗
║                    实践建议                                ║
╚════════════════════════════════════════════════════════════╝

1. 优先考虑Workflow
   • 如果任务可以用固定流程完成，优先使用Workflow
   • 更高效、更稳定、成本更低

2. 必要时使用Agent
   • 需要动态决策时
   • 需要工具调用时
   • 用户意图不明确时

3. 混合使用
   • 可以在Workflow中嵌入Agent
   • 也可以让Agent调用Workflow

4. 成本控制
   • Workflow: 固定成本（步骤数 × token）
   • Agent: 变动成本（迭代次数 × token）
   • 设置max_iterations限制Agent成本

5. 调试策略
   • Workflow: 检查每个步骤的输出
   • Agent: 启用verbose查看推理过程
"""

    print(comparison)


def demo_hybrid_approach():
    """演示混合方法"""
    print("\n" + "=" * 60)
    print("5. 混合方法 - Workflow + Agent")
    print("=" * 60)
    print("\n场景: 智能文档处理（Workflow处理 + Agent决策）")
    print("-" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Workflow部分：固定的文档处理流程
    extract_prompt = PromptTemplate(
        input_variables=["document"],
        template="提取以下文档的关键信息（人名、地点、日期）:\n\n{document}\n\n关键信息:"
    )
    extract_chain = LLMChain(llm=llm, prompt=extract_prompt, output_key="info")

    # Agent部分：根据提取的信息决定下一步
    def save_info(info: str) -> str:
        return f"✓ 信息已保存: {info[:50]}..."

    def send_notification(info: str) -> str:
        return f"✓ 通知已发送: {info[:50]}..."

    tools = [
        Tool(name="SaveInfo", func=save_info, description="保存信息"),
        Tool(name="SendNotification", func=send_notification, description="发送通知")
    ]

    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # 测试
    document = """
    会议纪要
    时间: 2026年1月25日
    地点: 北京办公室
    参与人: 张三、李四、王五

    讨论了新项目的启动计划，决定下周开始实施。
    """

    print(f"文档:\n{document}")
    print("\n步骤1: Workflow处理（提取信息）")
    print("-" * 60)

    # 先用Workflow提取信息
    extracted = extract_chain.invoke({"document": document})
    print(f"\n提取的信息:\n{extracted['info']}")

    print("\n步骤2: Agent决策（处理信息）")
    print("-" * 60)

    # 再用Agent决定如何处理
    agent_result = agent_executor.invoke({
        "input": f"根据以下信息，决定是保存还是发送通知:\n{extracted['info']}"
    })

    print(f"\nAgent决策: {agent_result['output']}")


def main():
    """主函数"""
    print("\n🔄 Workflow vs Agent 对比演示\n")

    try:
        # 1. Workflow - 固定步骤
        demo_workflow_fixed_steps()

        # 2. Workflow - 条件分支
        demo_workflow_conditional()

        # 3. Agent - 动态工具
        demo_agent_dynamic_tools()

        # 4. 对比分析
        comparison_analysis()

        # 5. 混合方法
        demo_hybrid_approach()

        print("\n" + "=" * 60)
        print("✅ 所有演示完成！")
        print("=" * 60)

        print("\n💡 核心要点:")
        print("1. Workflow适合固定流程，Agent适合动态决策")
        print("2. 优先考虑Workflow，必要时才用Agent")
        print("3. 可以混合使用两种方法")
        print("4. Workflow成本低、稳定；Agent灵活、强大")
        print("5. 根据具体场景选择合适的方案")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请检查API配置和网络连接")


if __name__ == "__main__":
    main()
