"""
OpenAI API 基础调用示例

学习目标：
1. 理解如何初始化OpenAI客户端
2. 学习基础的chat completion调用
3. 了解不同参数的作用
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def basic_chat():
    """最简单的对话示例"""
    print("=" * 50)
    print("示例1: 基础对话")
    print("=" * 50)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 使用的模型
        messages=[
            {"role": "user", "content": "你好，请用一句话介绍什么是AI Agent"}
        ]
    )

    # 提取回复内容
    answer = response.choices[0].message.content
    print(f"AI回复: {answer}\n")

    # 查看token使用情况
    print(f"Token使用: {response.usage.prompt_tokens} (输入) + "
          f"{response.usage.completion_tokens} (输出) = "
          f"{response.usage.total_tokens} (总计)")
    print()


def chat_with_system_message():
    """使用system message设定角色"""
    print("=" * 50)
    print("示例2: 使用System Message设定角色")
    print("=" * 50)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "你是一位资深的Python开发工程师，擅长用简洁的代码解决问题。"
            },
            {
                "role": "user",
                "content": "如何读取一个JSON文件？"
            }
        ],
        temperature=0.7,  # 控制随机性，0-2之间，越高越随机
        max_tokens=200    # 限制输出长度
    )

    answer = response.choices[0].message.content
    print(f"AI回复:\n{answer}\n")


def multi_turn_conversation():
    """多轮对话示例"""
    print("=" * 50)
    print("示例3: 多轮对话")
    print("=" * 50)

    # 对话历史
    messages = [
        {"role": "system", "content": "你是一个友好的助手。"},
        {"role": "user", "content": "我想学习Python"},
    ]

    # 第一轮
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    assistant_reply = response.choices[0].message.content
    print(f"用户: 我想学习Python")
    print(f"AI: {assistant_reply}\n")

    # 将AI的回复加入对话历史
    messages.append({"role": "assistant", "content": assistant_reply})

    # 第二轮 - AI能记住上下文
    messages.append({"role": "user", "content": "从哪里开始比较好？"})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    assistant_reply = response.choices[0].message.content
    print(f"用户: 从哪里开始比较好？")
    print(f"AI: {assistant_reply}\n")


def different_temperatures():
    """演示temperature参数的效果"""
    print("=" * 50)
    print("示例4: Temperature参数对比")
    print("=" * 50)

    prompt = "给我推荐一个Python项目名称"

    # Temperature = 0 (确定性输出)
    print("Temperature = 0 (更确定、一致):")
    for i in range(2):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50
        )
        print(f"  第{i+1}次: {response.choices[0].message.content}")

    print()

    # Temperature = 1.5 (更随机)
    print("Temperature = 1.5 (更随机、创意):")
    for i in range(2):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.5,
            max_tokens=50
        )
        print(f"  第{i+1}次: {response.choices[0].message.content}")

    print()


def streaming_response():
    """流式输出示例（像ChatGPT那样逐字显示）"""
    print("=" * 50)
    print("示例5: 流式输出")
    print("=" * 50)

    print("AI回复: ", end="", flush=True)

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "用一句话解释什么是机器学习"}
        ],
        stream=True  # 启用流式输出
    )

    # 逐块接收并打印
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")


def error_handling():
    """错误处理示例"""
    print("=" * 50)
    print("示例6: 错误处理")
    print("=" * 50)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "测试"}],
            max_tokens=1000000  # 故意设置一个不合理的值
        )
    except Exception as e:
        print(f"捕获到错误: {type(e).__name__}")
        print(f"错误信息: {str(e)}\n")
        print("正确的做法是设置合理的max_tokens值（如100-4000）")


if __name__ == "__main__":
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("错误: 未找到OPENAI_API_KEY环境变量")
        print("请先创建.env文件并设置API密钥")
        exit(1)

    print("\n🚀 OpenAI API 基础调用示例\n")

    try:
        # 运行所有示例
        basic_chat()
        chat_with_system_message()
        multi_turn_conversation()
        different_temperatures()
        streaming_response()
        error_handling()

        print("=" * 50)
        print("✅ 所有示例运行完成！")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("\n常见问题:")
        print("1. 检查API密钥是否正确")
        print("2. 检查网络连接")
        print("3. 检查是否需要设置代理")
