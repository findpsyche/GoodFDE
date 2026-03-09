"""
Anthropic Claude API 基础调用示例

学习目标：
1. 理解Anthropic API的调用方式
2. 对比OpenAI和Anthropic的API差异
3. 了解Claude模型的特点
"""

import os
from anthropic import Anthropic
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化Anthropic客户端
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def basic_chat():
    """最简单的对话示例"""
    print("=" * 50)
    print("示例1: 基础对话")
    print("=" * 50)

    message = client.messages.create(
        model="claude-3-5-haiku-20241022",  # 使用Haiku模型（快速且便宜）
        max_tokens=1024,  # Anthropic要求必须设置max_tokens
        messages=[
            {"role": "user", "content": "你好，请用一句话介绍什么是AI Agent"}
        ]
    )

    # 提取回复内容
    answer = message.content[0].text
    print(f"Claude回复: {answer}\n")

    # 查看token使用情况
    print(f"Token使用: {message.usage.input_tokens} (输入) + "
          f"{message.usage.output_tokens} (输出)")
    print()


def chat_with_system_prompt():
    """使用system prompt设定角色"""
    print("=" * 50)
    print("示例2: 使用System Prompt设定角色")
    print("=" * 50)

    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        system="你是一位资深的Python开发工程师，擅长用简洁的代码解决问题。",  # system是单独的参数
        messages=[
            {"role": "user", "content": "如何读取一个JSON文件？"}
        ],
        temperature=0.7
    )

    answer = message.content[0].text
    print(f"Claude回复:\n{answer}\n")


def multi_turn_conversation():
    """多轮对话示例"""
    print("=" * 50)
    print("示例3: 多轮对话")
    print("=" * 50)

    # 对话历史
    messages = [
        {"role": "user", "content": "我想学习Python"},
    ]

    # 第一轮
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        messages=messages
    )

    assistant_reply = response.content[0].text
    print(f"用户: 我想学习Python")
    print(f"Claude: {assistant_reply}\n")

    # 将Claude的回复加入对话历史
    messages.append({"role": "assistant", "content": assistant_reply})

    # 第二轮 - Claude能记住上下文
    messages.append({"role": "user", "content": "从哪里开始比较好？"})

    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        messages=messages
    )

    assistant_reply = response.content[0].text
    print(f"用户: 从哪里开始比较好？")
    print(f"Claude: {assistant_reply}\n")


def streaming_response():
    """流式输出示例"""
    print("=" * 50)
    print("示例4: 流式输出")
    print("=" * 50)

    print("Claude回复: ", end="", flush=True)

    with client.messages.stream(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "用一句话解释什么是机器学习"}
        ]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print("\n")


def compare_models():
    """对比不同Claude模型"""
    print("=" * 50)
    print("示例5: 对比不同模型")
    print("=" * 50)

    prompt = "用一句话解释量子计算"

    models = [
        "claude-3-5-haiku-20241022",   # 最快最便宜
        "claude-3-5-sonnet-20241022",  # 平衡性能和成本
    ]

    for model in models:
        print(f"\n模型: {model}")

        message = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        print(f"回复: {message.content[0].text}")
        print(f"Token: {message.usage.input_tokens} + {message.usage.output_tokens}")


def long_context_example():
    """长文本处理示例"""
    print("=" * 50)
    print("示例6: 长文本处理")
    print("=" * 50)

    # 模拟一个长文档
    long_text = """
    人工智能（Artificial Intelligence, AI）是计算机科学的一个分支，
    致力于创建能够执行通常需要人类智能的任务的系统。

    AI的主要领域包括：
    1. 机器学习：让计算机从数据中学习
    2. 自然语言处理：理解和生成人类语言
    3. 计算机视觉：理解和分析图像
    4. 机器人技术：创建能够与物理世界交互的智能系统

    近年来，深度学习的突破推动了AI的快速发展，
    特别是在图像识别、语音识别和自然语言处理等领域。
    """ * 10  # 重复10次模拟长文本

    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": f"请总结以下文本的核心要点：\n\n{long_text}"
            }
        ]
    )

    print(f"原文长度: {len(long_text)} 字符")
    print(f"输入tokens: {message.usage.input_tokens}")
    print(f"\nClaude总结:\n{message.content[0].text}\n")


def error_handling():
    """错误处理示例"""
    print("=" * 50)
    print("示例7: 错误处理")
    print("=" * 50)

    try:
        # 故意不设置max_tokens（必需参数）
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "测试"}]
            # 缺少max_tokens参数
        )
    except Exception as e:
        print(f"捕获到错误: {type(e).__name__}")
        print(f"错误信息: {str(e)}\n")
        print("Anthropic API要求必须设置max_tokens参数")


def api_comparison():
    """OpenAI vs Anthropic API对比"""
    print("=" * 50)
    print("OpenAI vs Anthropic API 主要差异")
    print("=" * 50)

    comparison = """
    1. 初始化方式:
       OpenAI:    client = OpenAI(api_key=...)
       Anthropic: client = Anthropic(api_key=...)

    2. 调用方法:
       OpenAI:    client.chat.completions.create(...)
       Anthropic: client.messages.create(...)

    3. System消息:
       OpenAI:    在messages列表中，role="system"
       Anthropic: 单独的system参数

    4. max_tokens:
       OpenAI:    可选参数
       Anthropic: 必需参数

    5. 响应格式:
       OpenAI:    response.choices[0].message.content
       Anthropic: message.content[0].text

    6. Token统计:
       OpenAI:    response.usage.prompt_tokens / completion_tokens
       Anthropic: message.usage.input_tokens / output_tokens

    7. 流式输出:
       OpenAI:    stream=True, 遍历chunk
       Anthropic: 使用with client.messages.stream()上下文管理器
    """

    print(comparison)


if __name__ == "__main__":
    # 检查API密钥
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("错误: 未找到ANTHROPIC_API_KEY环境变量")
        print("请先创建.env文件并设置API密钥")
        exit(1)

    print("\n🚀 Anthropic Claude API 基础调用示例\n")

    try:
        # 运行所有示例
        basic_chat()
        chat_with_system_prompt()
        multi_turn_conversation()
        streaming_response()
        compare_models()
        long_context_example()
        error_handling()
        api_comparison()

        print("=" * 50)
        print("✅ 所有示例运行完成！")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("\n常见问题:")
        print("1. 检查API密钥是否正确")
        print("2. 检查网络连接")
        print("3. 确保设置了max_tokens参数")
