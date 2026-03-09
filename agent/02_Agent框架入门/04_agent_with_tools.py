"""
完整的Agent示例 - 带多种工具
包括计算器、搜索、天气查询等功能
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain import hub
import requests
from datetime import datetime
import json

# 加载环境变量
load_dotenv()


# ============ 工具函数定义 ============

def calculator(expression: str) -> str:
    """
    计算数学表达式
    支持: +, -, *, /, **, (), 基本数学函数
    """
    try:
        # 安全的数学计算环境
        import math
        safe_dict = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow,
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
            'tan': math.tan, 'pi': math.pi, 'e': math.e
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


def search_wikipedia(query: str) -> str:
    """
    搜索维基百科
    返回查询主题的简短摘要
    """
    try:
        import wikipedia
        wikipedia.set_lang("zh")

        # 搜索并获取摘要
        try:
            summary = wikipedia.summary(query, sentences=3)
            return f"维基百科搜索结果:\n{summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            # 如果有歧义，返回第一个选项
            return f"找到多个相关主题: {', '.join(e.options[:5])}"
        except wikipedia.exceptions.PageError:
            return f"未找到关于 '{query}' 的信息"

    except ImportError:
        return "需要安装wikipedia库: pip install wikipedia"
    except Exception as e:
        return f"搜索错误: {str(e)}"


def get_current_time(query: str = "") -> str:
    """
    获取当前时间和日期
    """
    now = datetime.now()
    return f"""当前时间信息:
日期: {now.strftime('%Y年%m月%d日')}
时间: {now.strftime('%H:%M:%S')}
星期: {now.strftime('%A')}
"""


def get_weather(city: str) -> str:
    """
    获取城市天气信息（模拟）
    实际使用需要配置天气API
    """
    # 这里使用模拟数据
    # 实际应用中应该调用真实的天气API
    weather_data = {
        "北京": {"temp": 15, "condition": "晴", "humidity": 45},
        "上海": {"temp": 20, "condition": "多云", "humidity": 60},
        "深圳": {"temp": 25, "condition": "小雨", "humidity": 75},
        "成都": {"temp": 18, "condition": "阴", "humidity": 70},
    }

    if city in weather_data:
        data = weather_data[city]
        return f"""{city}天气:
温度: {data['temp']}°C
天气: {data['condition']}
湿度: {data['humidity']}%
"""
    else:
        return f"暂无{city}的天气数据（支持: 北京、上海、深圳、成都）"


def text_analyzer(text: str) -> str:
    """
    分析文本的基本信息
    """
    char_count = len(text)
    word_count = len(text.split())
    line_count = len(text.split('\n'))

    # 统计字符类型
    chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_count = sum(1 for c in text if c.isalpha() and c.isascii())
    digit_count = sum(1 for c in text if c.isdigit())

    return f"""文本分析结果:
总字符数: {char_count}
单词数: {word_count}
行数: {line_count}
中文字符: {chinese_count}
英文字符: {english_count}
数字字符: {digit_count}
"""


def file_writer(content: str) -> str:
    """
    将内容写入文件
    格式: filename.txt|content
    """
    try:
        if '|' not in content:
            return "格式错误。请使用: filename.txt|要写入的内容"

        filename, text = content.split('|', 1)
        filename = filename.strip()

        # 确保在output目录
        os.makedirs('output', exist_ok=True)
        filepath = os.path.join('output', filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text.strip())

        return f"✅ 内容已保存到: {filepath}"
    except Exception as e:
        return f"写入文件失败: {str(e)}"


# ============ 创建Agent ============

def create_multi_tool_agent():
    """创建带多种工具的Agent"""

    # 初始化LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=1000
    )

    # 定义工具列表
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="""用于数学计算。支持基本运算(+,-,*,/,**)和数学函数(sqrt,sin,cos等)。
输入应该是一个数学表达式，例如: '2+2', '10*5', 'sqrt(16)', 'pi*2'"""
        ),
        Tool(
            name="Wikipedia",
            func=search_wikipedia,
            description="""搜索维基百科获取知识。当需要查询人物、地点、概念、历史事件等信息时使用。
输入应该是要搜索的主题，例如: '人工智能', '北京', '爱因斯坦'"""
        ),
        Tool(
            name="CurrentTime",
            func=get_current_time,
            description="""获取当前的日期和时间。当用户询问现在几点、今天日期、星期几等问题时使用。
不需要输入参数。"""
        ),
        Tool(
            name="Weather",
            func=get_weather,
            description="""查询城市天气信息。支持北京、上海、深圳、成都。
输入应该是城市名称，例如: '北京', '上海'"""
        ),
        Tool(
            name="TextAnalyzer",
            func=text_analyzer,
            description="""分析文本的统计信息，包括字符数、单词数、中英文字符等。
输入应该是要分析的文本。"""
        ),
        Tool(
            name="FileWriter",
            func=file_writer,
            description="""将内容保存到文件。格式: 'filename.txt|要保存的内容'
例如: 'note.txt|这是要保存的内容'"""
        )
    ]

    # 创建memory
    memory = ConversationBufferWindowMemory(
        k=5,  # 保留最近5轮对话
        memory_key="chat_history",
        return_messages=True
    )

    # 获取prompt模板
    prompt = hub.pull("hwchase17/react")

    # 创建agent
    agent = create_react_agent(llm, tools, prompt)

    # 创建agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        early_stopping_method="generate"
    )

    return agent_executor


# ============ 测试场景 ============

def test_scenarios():
    """测试不同场景"""

    agent = create_multi_tool_agent()

    test_cases = [
        {
            "name": "数学计算",
            "query": "计算 (25 + 15) * 3 - 10"
        },
        {
            "name": "知识查询",
            "query": "介绍一下LangChain是什么"
        },
        {
            "name": "时间查询",
            "query": "现在几点了？今天星期几？"
        },
        {
            "name": "天气查询",
            "query": "北京今天天气怎么样？"
        },
        {
            "name": "文本分析",
            "query": "分析这段文本: 'Hello World! 你好世界！123'"
        },
        {
            "name": "复杂任务",
            "query": "计算100的平方根，然后搜索关于'平方根'的信息，最后把结果保存到result.txt"
        }
    ]

    print("\n" + "=" * 60)
    print("🤖 多工具Agent测试")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}: {test['name']}")
        print(f"{'='*60}")
        print(f"问题: {test['query']}")
        print("-" * 60)

        try:
            result = agent.invoke({"input": test['query']})
            print(f"\n✅ 最终答案:\n{result['output']}")
        except Exception as e:
            print(f"\n❌ 错误: {e}")

        print()


def interactive_mode():
    """交互模式"""

    agent = create_multi_tool_agent()

    print("\n" + "=" * 60)
    print("🤖 多工具Agent - 交互模式")
    print("=" * 60)
    print("\n可用工具:")
    print("  • Calculator - 数学计算")
    print("  • Wikipedia - 知识查询")
    print("  • CurrentTime - 时间查询")
    print("  • Weather - 天气查询")
    print("  • TextAnalyzer - 文本分析")
    print("  • FileWriter - 文件保存")
    print("\n输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出']:
                print("\n👋 再见！")
                break

            if not user_input:
                continue

            print("\nAgent思考中...")
            print("-" * 60)

            result = agent.invoke({"input": user_input})

            print(f"\n🤖 Agent: {result['output']}")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


def main():
    """主函数"""
    print("\n🚀 LangChain多工具Agent演示\n")

    print("选择模式:")
    print("1. 自动测试模式（运行预设测试）")
    print("2. 交互模式（手动输入问题）")

    try:
        choice = input("\n请选择 (1/2): ").strip()

        if choice == "1":
            test_scenarios()
        elif choice == "2":
            interactive_mode()
        else:
            print("无效选择，运行自动测试模式...")
            test_scenarios()

        print("\n" + "=" * 60)
        print("✅ 演示完成！")
        print("=" * 60)

        print("\n💡 关键要点:")
        print("1. Agent可以集成多种工具")
        print("2. Agent会自动选择合适的工具")
        print("3. 可以组合多个工具完成复杂任务")
        print("4. Memory让Agent记住对话历史")
        print("5. 需要为每个工具提供清晰的描述")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请检查:")
        print("1. API密钥配置")
        print("2. 依赖包安装: pip install wikipedia")
        print("3. 网络连接")


if __name__ == "__main__":
    main()
