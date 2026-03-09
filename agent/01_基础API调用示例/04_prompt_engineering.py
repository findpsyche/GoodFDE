"""
Prompt Engineering 技巧示例

学习目标：
1. 理解什么是好的prompt
2. 学习常见的prompt工程技巧
3. 对比不同prompt的效果
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def compare_prompts(bad_prompt, good_prompt, task_name):
    """对比两个prompt的效果"""
    print("=" * 60)
    print(f"📝 任务: {task_name}")
    print("=" * 60)

    # 差的prompt
    print("\n❌ 差的Prompt:")
    print(f"   \"{bad_prompt}\"")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": bad_prompt}],
        temperature=0.7,
        max_tokens=200
    )
    print(f"\n回复:\n{response.choices[0].message.content}\n")

    # 好的prompt
    print("✅ 好的Prompt:")
    print(f"   \"{good_prompt}\"")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": good_prompt}],
        temperature=0.7,
        max_tokens=200
    )
    print(f"\n回复:\n{response.choices[0].message.content}\n")


def technique_1_be_specific():
    """技巧1: 明确具体"""
    compare_prompts(
        bad_prompt="写一个Python函数",
        good_prompt="写一个Python函数，接收一个整数列表作为参数，返回列表中所有偶数的和。函数名为sum_even_numbers，包含类型注解和docstring。",
        task_name="明确具体的指令"
    )


def technique_2_provide_context():
    """技巧2: 提供上下文"""
    compare_prompts(
        bad_prompt="这个代码有什么问题？\ndef add(a, b):\n    return a + b",
        good_prompt="""我正在开发一个计算器应用，需要处理用户输入的数字。
下面这个函数用于加法运算，但用户可能输入非数字类型。
请帮我找出潜在问题并给出改进建议：

def add(a, b):
    return a + b
""",
        task_name="提供充分的上下文"
    )


def technique_3_step_by_step():
    """技巧3: 要求分步思考"""
    print("=" * 60)
    print("📝 任务: 要求分步思考（Chain of Thought）")
    print("=" * 60)

    # 不要求分步
    print("\n❌ 不要求分步:")
    prompt1 = "一个数字加上它的两倍等于15，这个数字是多少？"
    print(f"   \"{prompt1}\"")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt1}],
        temperature=0
    )
    print(f"\n回复:\n{response.choices[0].message.content}\n")

    # 要求分步
    print("✅ 要求分步思考:")
    prompt2 = "一个数字加上它的两倍等于15，这个数字是多少？请一步一步思考并解释你的推理过程。"
    print(f"   \"{prompt2}\"")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt2}],
        temperature=0
    )
    print(f"\n回复:\n{response.choices[0].message.content}\n")


def technique_4_use_examples():
    """技巧4: 提供示例（Few-shot Learning）"""
    print("=" * 60)
    print("📝 任务: 提供示例（Few-shot）")
    print("=" * 60)

    # 不提供示例
    print("\n❌ 不提供示例（Zero-shot）:")
    prompt1 = "将以下句子改写成更正式的语气：今天天气真好啊"
    print(f"   \"{prompt1}\"")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt1}],
        temperature=0.7
    )
    print(f"\n回复:\n{response.choices[0].message.content}\n")

    # 提供示例
    print("✅ 提供示例（Few-shot）:")
    prompt2 = """将以下句子改写成更正式的语气。

示例1:
输入: 这个东西真不错
输出: 此物品质量上乘

示例2:
输入: 我觉得这个主意挺好的
输出: 本人认为该提议颇具价值

现在请改写:
输入: 今天天气真好啊
输出:"""
    print(f"   \"{prompt2}\"")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt2}],
        temperature=0.7
    )
    print(f"\n回复:\n{response.choices[0].message.content}\n")


def technique_5_specify_format():
    """技巧5: 指定输出格式"""
    print("=" * 60)
    print("📝 任务: 指定输出格式")
    print("=" * 60)

    # 不指定格式
    print("\n❌ 不指定格式:")
    prompt1 = "给我3个Python学习资源"
    print(f"   \"{prompt1}\"")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt1}],
        temperature=0.7
    )
    print(f"\n回复:\n{response.choices[0].message.content}\n")

    # 指定JSON格式
    print("✅ 指定JSON格式:")
    prompt2 = """给我3个Python学习资源，以JSON格式返回，包含name、url、description字段。

格式示例:
{
  "resources": [
    {
      "name": "资源名称",
      "url": "https://example.com",
      "description": "简短描述"
    }
  ]
}"""
    print(f"   \"{prompt2}\"")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt2}],
        temperature=0.7
    )
    print(f"\n回复:\n{response.choices[0].message.content}\n")


def technique_6_role_playing():
    """技巧6: 角色扮演"""
    print("=" * 60)
    print("📝 任务: 角色扮演")
    print("=" * 60)

    # 普通提问
    print("\n❌ 普通提问:")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "如何优化Python代码性能？"}
        ],
        temperature=0.7,
        max_tokens=200
    )
    print(f"回复:\n{response.choices[0].message.content}\n")

    # 角色扮演
    print("✅ 角色扮演:")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "你是一位有20年经验的Python性能优化专家，曾在Google工作，擅长用实际案例解释复杂概念。"
            },
            {
                "role": "user",
                "content": "如何优化Python代码性能？请给我3个最重要的建议。"
            }
        ],
        temperature=0.7,
        max_tokens=200
    )
    print(f"回复:\n{response.choices[0].message.content}\n")


def technique_7_constraints():
    """技巧7: 设置约束条件"""
    print("=" * 60)
    print("📝 任务: 设置约束条件")
    print("=" * 60)

    # 无约束
    print("\n❌ 无约束:")
    prompt1 = "解释什么是机器学习"
    print(f"   \"{prompt1}\"")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt1}],
        temperature=0.7
    )
    print(f"\n回复:\n{response.choices[0].message.content}\n")

    # 有约束
    print("✅ 有约束:")
    prompt2 = """用一句话（不超过30个字）向一个10岁小孩解释什么是机器学习。
要求：
1. 使用简单的日常比喻
2. 避免技术术语
3. 确保小孩能理解"""
    print(f"   \"{prompt2}\"")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt2}],
        temperature=0.7
    )
    print(f"\n回复:\n{response.choices[0].message.content}\n")


def technique_8_negative_prompting():
    """技巧8: 负面提示（告诉AI不要做什么）"""
    print("=" * 60)
    print("📝 任务: 负面提示")
    print("=" * 60)

    # 无负面提示
    print("\n❌ 无负面提示:")
    prompt1 = "给我一个Python Web框架推荐"
    print(f"   \"{prompt1}\"")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt1}],
        temperature=0.7,
        max_tokens=150
    )
    print(f"\n回复:\n{response.choices[0].message.content}\n")

    # 有负面提示
    print("✅ 有负面提示:")
    prompt2 = """给我一个Python Web框架推荐。

要求：
- 只推荐一个框架
- 不要列举多个选项
- 不要说"这取决于你的需求"
- 直接给出明确推荐和理由"""
    print(f"   \"{prompt2}\"")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt2}],
        temperature=0.7,
        max_tokens=150
    )
    print(f"\n回复:\n{response.choices[0].message.content}\n")


def prompt_template_example():
    """实用的Prompt模板"""
    print("=" * 60)
    print("📝 实用Prompt模板")
    print("=" * 60)

    template = """
任务: {task}
上下文: {context}
要求:
{requirements}
输出格式: {format}
约束条件: {constraints}
"""

    print("\n模板结构:")
    print(template)

    print("\n实际应用示例:")
    actual_prompt = """
任务: 代码审查
上下文: 这是一个用户注册功能的Python代码，用于Web应用
要求:
1. 检查安全性问题
2. 检查代码质量
3. 提供改进建议
输出格式: 分点列出，每点包含问题描述和改进方案
约束条件: 只关注最重要的3个问题

代码:
def register_user(username, password):
    users[username] = password
    return "Success"
"""
    print(actual_prompt)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": actual_prompt}],
        temperature=0.7
    )
    print(f"\nAI回复:\n{response.choices[0].message.content}\n")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("🎯 Prompt Engineering 技巧示例")
    print("=" * 60 + "\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误: 未找到OPENAI_API_KEY")
        return

    try:
        technique_1_be_specific()
        technique_2_provide_context()
        technique_3_step_by_step()
        technique_4_use_examples()
        technique_5_specify_format()
        technique_6_role_playing()
        technique_7_constraints()
        technique_8_negative_prompting()
        prompt_template_example()

        print("=" * 60)
        print("✅ 所有示例运行完成！")
        print("=" * 60)
        print("\n💡 Prompt Engineering 核心原则:")
        print("1. 明确具体 - 清楚说明你想要什么")
        print("2. 提供上下文 - 给AI足够的背景信息")
        print("3. 分步思考 - 引导AI逐步推理")
        print("4. 使用示例 - 展示期望的输出格式")
        print("5. 设置约束 - 限定输出的范围和格式")
        print("6. 角色扮演 - 让AI扮演特定角色")
        print("7. 负面提示 - 明确不想要的内容")
        print("8. 迭代优化 - 根据结果不断改进prompt\n")

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")


if __name__ == "__main__":
    main()
