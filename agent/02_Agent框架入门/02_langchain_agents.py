"""
LangChain Agent和Tools演示
展示如何创建Agent并使用工具
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.prompts import PromptTemplate
from langchain import hub
import datetime

# 加载环境变量
load_dotenv()


# ============ 自定义工具函数 ============

def get_current_time(query: str) -> str:
    """获取当前时间"""
    now = datetime.datetime.now()
    return f"当前时间是: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def calculate(expression: str) -> str:
    """
    计算数学表达式
    支持基本的加减乘除运算
    """
    try:
        # 安全的数学计算
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


def get_word_length(text: str) -> str:
    """获取文本长度"""
    length = len(text)
    return f"文本 '{text}' 的长度是 {length} 个字符"


def reverse_string(text: str) -> str:
    """反转字符串"""
    reversed_text = text[::-1]
    return f"'{text}' 反转后是 '{reversed_text}'"


# ============ Agent演示 ============

def demo_basic_agent():
    """演示基础Agent"""
    print("=" * 50)
    print("1. 基础Agent演示")
    print("=" * 50)

    # 初始化LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 定义工具
    tools = [
        Tool(
            name="CurrentTime",
            func=get_current_time,
            description="获取当前的日期和时间。当用户询问现在几点、今天日期等问题时使用。"
        ),
        Tool(
            name="Calculator",
            func=calculate,
            description="计算数学表达式。输入应该是一个有效的数学表达式，例如: 2+2, 10*5, 100/4"
        ),
        Tool(
            name="TextLength",
            func=get_word_length,
            description="获取文本的字符长度。输入应该是一个字符串。"
        )
    ]

    # 获取ReAct prompt模板
    prompt = hub.pull("hwchase17/react")

    # 创建Agent
    agent = create_react_agent(llm, tools, prompt)

    # 创建Agent执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )

    # 测试问题
    questions = [
        "现在几点了？",
        "计算 25 * 4 + 10",
        "'Hello World' 这个字符串有多长？"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 50)
        try:
            result = agent_executor.invoke({"input": question})
            print(f"\n最终答案: {result['output']}")
        except Exception as e:
            print(f"错误: {e}")
        print()


def demo_agent_with_custom_prompt():
    """演示自定义Prompt的Agent"""
    print("\n" + "=" * 50)
    print("2. 自定义Prompt的Agent")
    print("=" * 50)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 定义工具
    tools = [
        Tool(
            name="Calculator",
            func=calculate,
            description="用于数学计算"
        ),
        Tool(
            name="StringReverse",
            func=reverse_string,
            description="反转字符串"
        )
    ]

    # 自定义ReAct prompt
    custom_prompt = PromptTemplate.from_template("""
你是一个有用的AI助手，可以使用工具来回答问题。

你可以使用以下工具:
{tools}

工具名称: {tool_names}

使用以下格式:

Question: 用户的问题
Thought: 你应该思考要做什么
Action: 要使用的工具，应该是 [{tool_names}] 中的一个
Action Input: 工具的输入
Observation: 工具的输出
... (这个 Thought/Action/Action Input/Observation 可以重复N次)
Thought: 我现在知道最终答案了
Final Answer: 对原始问题的最终答案

开始!

Question: {input}
Thought: {agent_scratchpad}
""")

    # 创建Agent
    agent = create_react_agent(llm, tools, custom_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # 测试
    question = "计算 15 * 3，然后把结果反转"
    print(f"\n问题: {question}")
    print("-" * 50)

    try:
        result = agent_executor.invoke({"input": question})
        print(f"\n最终答案: {result['output']}")
    except Exception as e:
        print(f"错误: {e}")


def demo_agent_reasoning():
    """演示Agent的推理过程"""
    print("\n" + "=" * 50)
    print("3. Agent推理过程演示")
    print("=" * 50)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 定义工具
    tools = [
        Tool(
            name="CurrentTime",
            func=get_current_time,
            description="获取当前时间"
        ),
        Tool(
            name="Calculator",
            func=calculate,
            description="计算数学表达式"
        )
    ]

    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True  # 返回中间步骤
    )

    # 复杂问题
    question = "如果现在是下午3点，那么5小时后是几点？"
    print(f"\n问题: {question}")
    print("-" * 50)

    try:
        result = agent_executor.invoke({"input": question})

        print("\n" + "=" * 50)
        print("推理步骤分析:")
        print("=" * 50)

        for i, (action, observation) in enumerate(result['intermediate_steps'], 1):
            print(f"\n步骤 {i}:")
            print(f"  工具: {action.tool}")
            print(f"  输入: {action.tool_input}")
            print(f"  输出: {observation}")

        print(f"\n最终答案: {result['output']}")

    except Exception as e:
        print(f"错误: {e}")


def demo_error_handling():
    """演示错误处理"""
    print("\n" + "=" * 50)
    print("4. 错误处理演示")
    print("=" * 50)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tools = [
        Tool(
            name="Calculator",
            func=calculate,
            description="计算数学表达式"
        )
    ]

    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,  # 自动处理解析错误
        max_iterations=3  # 限制最大迭代次数
    )

    # 测试错误情况
    test_cases = [
        "计算 10 除以 0",  # 数学错误
        "今天天气怎么样？",  # 没有合适的工具
    ]

    for question in test_cases:
        print(f"\n问题: {question}")
        print("-" * 50)
        try:
            result = agent_executor.invoke({"input": question})
            print(f"结果: {result['output']}")
        except Exception as e:
            print(f"捕获错误: {e}")


def main():
    """主函数"""
    print("\n🤖 LangChain Agent和Tools演示\n")

    try:
        # 1. 基础Agent
        demo_basic_agent()

        # 2. 自定义Prompt
        demo_agent_with_custom_prompt()

        # 3. 推理过程
        demo_agent_reasoning()

        # 4. 错误处理
        demo_error_handling()

        print("\n" + "=" * 50)
        print("✅ 所有演示完成！")
        print("=" * 50)

        print("\n💡 关键要点:")
        print("1. Agent = LLM + Tools + 推理循环")
        print("2. ReAct模式: Reasoning (推理) + Acting (行动)")
        print("3. 工具描述很重要，影响Agent的选择")
        print("4. 需要设置max_iterations防止无限循环")
        print("5. handle_parsing_errors帮助处理格式错误")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请检查API配置和网络连接")


if __name__ == "__main__":
    main()
