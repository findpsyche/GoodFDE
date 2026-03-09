"""
ReAct模式示例 - Reasoning and Acting

ReAct模式是一种将推理(Reasoning)和行动(Acting)结合的Agent设计模式。
Agent在每一步都会：
1. Thought: 思考当前状态和下一步行动
2. Action: 选择并执行一个工具
3. Observation: 观察工具执行的结果
4. 重复上述过程，直到得出最终答案

核心设计原则：
- 验证驱动：每一步都有日志和验证
- 闭环思维：观察结果，决定是否继续
- 降低认知负荷：清晰的工具描述和示例
"""

import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.callbacks import StdOutCallbackHandler
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


# ============================================================================
# 自定义工具定义
# ============================================================================

def search_web(query: str) -> str:
    """
    模拟网页搜索工具

    Args:
        query: 搜索关键词（1-100个字符）

    Returns:
        搜索结果的文本描述

    注意事项：
        - 这是一个模拟工具，实际应用中应该调用真实的搜索API
        - 真实场景中需要处理速率限制、错误重试等
    """
    logger.info(f"🔍 搜索: {query}")

    # 模拟搜索结果
    results = {
        "python": "Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。",
        "agent": "AI Agent是能够自主感知环境、做出决策并执行任务的智能系统。",
        "react": "ReAct是一种结合推理和行动的Agent设计模式，由Google提出。"
    }

    # 简单的关键词匹配
    for key, value in results.items():
        if key.lower() in query.lower():
            return f"搜索结果：{value}"

    return f"搜索结果：关于'{query}'的信息..."


def calculate(expression: str) -> str:
    """
    计算数学表达式

    Args:
        expression: 数学表达式（如 "2 + 3 * 4"）

    Returns:
        计算结果

    注意事项：
        - 只支持基本的数学运算
        - 使用eval()有安全风险，生产环境应使用更安全的方法
    """
    logger.info(f"🔢 计算: {expression}")

    try:
        # 安全检查：只允许数字和基本运算符
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"

        result = eval(expression)
        return f"计算结果：{result}"

    except Exception as e:
        logger.error(f"计算错误: {e}")
        return f"计算错误：{str(e)}"


def get_current_time() -> str:
    """
    获取当前时间

    Returns:
        当前时间的字符串表示
    """
    logger.info("⏰ 获取当前时间")
    now = datetime.now()
    return f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"


# ============================================================================
# 创建工具列表
# ============================================================================

tools = [
    Tool(
        name="search_web",
        func=search_web,
        description="""
        搜索网页获取信息。

        使用场景：
        - 需要查找最新信息
        - 需要了解某个概念或技术

        输入：搜索关键词（字符串）
        输出：搜索结果的文本描述

        示例：
        - 输入: "Python最佳实践"
        - 输出: "搜索结果：Python最佳实践包括..."
        """
    ),
    Tool(
        name="calculate",
        func=calculate,
        description="""
        计算数学表达式。

        使用场景：
        - 需要进行数学计算
        - 需要处理数字运算

        输入：数学表达式（字符串，如 "2 + 3 * 4"）
        输出：计算结果

        示例：
        - 输入: "100 * 0.8"
        - 输出: "计算结果：80.0"
        """
    ),
    Tool(
        name="get_current_time",
        func=get_current_time,
        description="""
        获取当前时间。

        使用场景：
        - 需要知道当前时间
        - 需要时间戳

        输入：无
        输出：当前时间的字符串

        示例：
        - 输出: "当前时间：2026-02-06 10:30:00"
        """
    )
]


# ============================================================================
# 创建ReAct Agent
# ============================================================================

def create_react_agent_executor(verbose: bool = True) -> AgentExecutor:
    """
    创建ReAct Agent执行器

    Args:
        verbose: 是否显示详细日志

    Returns:
        AgentExecutor实例
    """
    # 选择模型
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0,
            verbose=verbose
        )
        logger.info("✅ 使用OpenAI模型")
    elif os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            temperature=0,
            verbose=verbose
        )
        logger.info("✅ 使用Anthropic模型")
    else:
        raise ValueError("请设置OPENAI_API_KEY或ANTHROPIC_API_KEY环境变量")

    # ReAct Prompt模板
    react_prompt = PromptTemplate.from_template("""
你是一个有用的AI助手，能够使用工具来回答问题。

你可以使用以下工具：
{tools}

工具描述：
{tool_names}

请使用以下格式：

Question: 用户的问题
Thought: 你应该思考要做什么
Action: 要使用的工具，必须是 [{tool_names}] 中的一个
Action Input: 工具的输入
Observation: 工具的输出
... (这个 Thought/Action/Action Input/Observation 可以重复N次)
Thought: 我现在知道最终答案了
Final Answer: 对原始问题的最终答案

重要规则：
1. 必须严格遵循上述格式
2. Action必须是提供的工具之一
3. 每次只能使用一个工具
4. 如果工具返回错误，思考如何解决
5. 最终答案必须以"Final Answer:"开头

开始！

Question: {input}
Thought: {agent_scratchpad}
""")

    # 创建agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )

    # 创建executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=10,  # 最大迭代次数，避免无限循环
        max_execution_time=60,  # 最大执行时间（秒）
        handle_parsing_errors=True,  # 处理解析错误
        return_intermediate_steps=True  # 返回中间步骤
    )

    return agent_executor


# ============================================================================
# 示例函数
# ============================================================================

def example_1_basic():
    """示例1：基础ReAct模式"""
    print("\n" + "="*50)
    print("示例1: ReAct模式 - 基础")
    print("="*50 + "\n")

    agent = create_react_agent_executor(verbose=True)

    question = "Python是什么？它有什么特点？"
    print(f"问题: {question}\n")

    try:
        result = agent.invoke({"input": question})

        print("\n" + "-"*50)
        print("最终答案:")
        print(result["output"])
        print("-"*50)

        # 显示中间步骤
        print("\n中间步骤:")
        for i, step in enumerate(result["intermediate_steps"], 1):
            action, observation = step
            print(f"\n步骤 {i}:")
            print(f"  工具: {action.tool}")
            print(f"  输入: {action.tool_input}")
            print(f"  输出: {observation}")

    except Exception as e:
        logger.error(f"执行失败: {e}")
        print(f"\n❌ 执行失败: {e}")


def example_2_calculation():
    """示例2：使用计算工具"""
    print("\n" + "="*50)
    print("示例2: ReAct模式 - 计算")
    print("="*50 + "\n")

    agent = create_react_agent_executor(verbose=True)

    question = "如果一个商品原价100元，打8折后再减10元，最终价格是多少？"
    print(f"问题: {question}\n")

    try:
        result = agent.invoke({"input": question})

        print("\n" + "-"*50)
        print("最终答案:")
        print(result["output"])
        print("-"*50)

    except Exception as e:
        logger.error(f"执行失败: {e}")
        print(f"\n❌ 执行失败: {e}")


def example_3_multi_tool():
    """示例3：使用多个工具"""
    print("\n" + "="*50)
    print("示例3: ReAct模式 - 多工具协作")
    print("="*50 + "\n")

    agent = create_react_agent_executor(verbose=True)

    question = "现在是什么时间？如果从现在开始工作8小时，结束时间是几点？"
    print(f"问题: {question}\n")

    try:
        result = agent.invoke({"input": question})

        print("\n" + "-"*50)
        print("最终答案:")
        print(result["output"])
        print("-"*50)

    except Exception as e:
        logger.error(f"执行失败: {e}")
        print(f"\n❌ 执行失败: {e}")


def example_4_error_handling():
    """示例4：错误处理"""
    print("\n" + "="*50)
    print("示例4: ReAct模式 - 错误处理")
    print("="*50 + "\n")

    agent = create_react_agent_executor(verbose=True)

    # 故意使用一个可能导致错误的问题
    question = "计算 10 除以 0 的结果"
    print(f"问题: {question}\n")

    try:
        result = agent.invoke({"input": question})

        print("\n" + "-"*50)
        print("最终答案:")
        print(result["output"])
        print("-"*50)

    except Exception as e:
        logger.error(f"执行失败: {e}")
        print(f"\n❌ 执行失败: {e}")
        print("\n💡 这个例子展示了Agent如何处理工具执行错误")


# ============================================================================
# 验证和测试
# ============================================================================

def verify_agent_quality(agent: AgentExecutor, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    验证Agent的质量

    Args:
        agent: Agent执行器
        test_cases: 测试用例列表

    Returns:
        验证结果
    """
    print("\n" + "="*50)
    print("Agent质量验证")
    print("="*50 + "\n")

    results = {
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "details": []
    }

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}/{len(test_cases)}")
        print(f"问题: {test_case['question']}")

        try:
            result = agent.invoke({"input": test_case["question"]})
            answer = result["output"]

            # 简单的验证：检查答案是否包含预期关键词
            expected_keywords = test_case.get("expected_keywords", [])
            passed = all(keyword.lower() in answer.lower() for keyword in expected_keywords)

            if passed:
                results["passed"] += 1
                print("✅ 通过")
            else:
                results["failed"] += 1
                print("❌ 失败")
                print(f"   预期包含: {expected_keywords}")
                print(f"   实际答案: {answer[:100]}...")

            results["details"].append({
                "question": test_case["question"],
                "answer": answer,
                "passed": passed
            })

        except Exception as e:
            results["failed"] += 1
            print(f"❌ 执行错误: {e}")
            results["details"].append({
                "question": test_case["question"],
                "error": str(e),
                "passed": False
            })

    # 打印总结
    print("\n" + "="*50)
    print("验证总结")
    print("="*50)
    print(f"总计: {results['total']}")
    print(f"通过: {results['passed']}")
    print(f"失败: {results['failed']}")
    print(f"通过率: {results['passed']/results['total']*100:.1f}%")

    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("\n" + "="*70)
    print("ReAct模式示例 - Reasoning and Acting")
    print("="*70)
    print("\n核心设计原则：")
    print("  ✅ 验证驱动：每一步都有日志和验证")
    print("  ✅ 闭环思维：观察结果，决定是否继续")
    print("  ✅ 降低认知负荷：清晰的工具描述和示例")
    print("\n" + "="*70 + "\n")

    # 检查API密钥
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("❌ 错误：请设置OPENAI_API_KEY或ANTHROPIC_API_KEY环境变量")
        print("\n请按照以下步骤配置：")
        print("1. 复制 .env.example 为 .env")
        print("2. 在 .env 文件中填入你的API密钥")
        print("3. 重新运行此脚本")
        return

    try:
        # 运行示例
        example_1_basic()

        input("\n按Enter继续下一个示例...")
        example_2_calculation()

        input("\n按Enter继续下一个示例...")
        example_3_multi_tool()

        input("\n按Enter继续下一个示例...")
        example_4_error_handling()

        # 质量验证
        input("\n按Enter运行质量验证...")

        agent = create_react_agent_executor(verbose=False)
        test_cases = [
            {
                "question": "Python是什么？",
                "expected_keywords": ["Python", "编程"]
            },
            {
                "question": "计算 50 * 2",
                "expected_keywords": ["100"]
            },
            {
                "question": "现在几点了？",
                "expected_keywords": ["时间", "2026"]
            }
        ]

        verify_agent_quality(agent, test_cases)

        print("\n" + "="*70)
        print("✅ 所有示例运行完成！")
        print("="*70)
        print("\n💡 学习要点：")
        print("  1. ReAct模式通过Thought-Action-Observation循环工作")
        print("  2. 每一步都有明确的推理过程")
        print("  3. 工具调用基于Agent的思考")
        print("  4. 可以处理多轮迭代和错误情况")
        print("\n📝 下一步：")
        print("  - 尝试修改问题，观察Agent的行为")
        print("  - 添加新的工具，扩展Agent的能力")
        print("  - 在学习笔记中记录你的发现")
        print("\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)
        print(f"\n❌ 程序执行失败: {e}")


if __name__ == "__main__":
    main()
