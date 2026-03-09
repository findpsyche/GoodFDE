"""
Token计数和成本估算工具

学习目标：
1. 理解Token的概念
2. 学习如何计算Token数量
3. 估算API调用成本
"""

import os
import tiktoken
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class TokenCounter:
    """Token计数和成本估算工具"""

    # 价格表（每百万tokens的价格，单位：美元）
    # 数据来源：2024年价格，实际使用时请查看官网最新价格
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    }

    def __init__(self):
        """初始化Token计数器"""
        # OpenAI使用tiktoken库
        self.gpt_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def count_tokens_openai(self, text, model="gpt-3.5-turbo"):
        """
        计算OpenAI模型的token数量

        Args:
            text: 要计算的文本
            model: 模型名称

        Returns:
            token数量
        """
        try:
            encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            encoder = self.gpt_encoder

        return len(encoder.encode(text))

    def count_tokens_anthropic(self, text):
        """
        估算Anthropic模型的token数量
        注意：这是近似值，Anthropic使用不同的tokenizer

        Args:
            text: 要计算的文本

        Returns:
            估算的token数量
        """
        # 使用OpenAI的tokenizer作为近似
        # 实际Anthropic的token数可能略有不同
        return self.count_tokens_openai(text)

    def estimate_cost(self, input_tokens, output_tokens, model):
        """
        估算API调用成本

        Args:
            input_tokens: 输入token数
            output_tokens: 输出token数
            model: 模型名称

        Returns:
            成本（美元）
        """
        if model not in self.PRICING:
            return None

        pricing = self.PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def analyze_conversation(self, messages, model):
        """
        分析对话的token使用情况

        Args:
            messages: 对话消息列表
            model: 模型名称

        Returns:
            分析结果字典
        """
        total_tokens = 0

        for msg in messages:
            content = msg.get("content", "")
            tokens = self.count_tokens_openai(content, model)
            total_tokens += tokens

        return {
            "total_tokens": total_tokens,
            "message_count": len(messages),
            "avg_tokens_per_message": total_tokens / len(messages) if messages else 0
        }


def demo_basic_counting():
    """演示基础token计数"""
    print("=" * 60)
    print("示例1: 基础Token计数")
    print("=" * 60)

    counter = TokenCounter()

    texts = [
        "Hello, world!",
        "你好，世界！",
        "This is a longer sentence with more words to demonstrate token counting.",
        "这是一个更长的中文句子，用来演示token计数的工作原理。",
    ]

    for text in texts:
        tokens = counter.count_tokens_openai(text)
        print(f"\n文本: {text}")
        print(f"字符数: {len(text)}")
        print(f"Token数: {tokens}")
        print(f"比例: {tokens/len(text):.2f} tokens/字符")


def demo_cost_estimation():
    """演示成本估算"""
    print("\n" + "=" * 60)
    print("示例2: 成本估算")
    print("=" * 60)

    counter = TokenCounter()

    # 模拟一次API调用
    input_text = "请用Python写一个快速排序算法"
    output_text = """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
"""

    input_tokens = counter.count_tokens_openai(input_text)
    output_tokens = counter.count_tokens_openai(output_text)

    print(f"\n输入文本: {input_text}")
    print(f"输入tokens: {input_tokens}")
    print(f"\n输出tokens: {output_tokens}")

    print("\n不同模型的成本对比:")
    print("-" * 60)

    models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ]

    for model in models:
        cost = counter.estimate_cost(input_tokens, output_tokens, model)
        if cost:
            print(f"{model:35s} ${cost:.6f} (¥{cost*7:.4f})")


def demo_conversation_analysis():
    """演示对话分析"""
    print("\n" + "=" * 60)
    print("示例3: 对话Token分析")
    print("=" * 60)

    counter = TokenCounter()

    # 模拟一个对话
    conversation = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "什么是机器学习？"},
        {"role": "assistant", "content": "机器学习是人工智能的一个分支，它让计算机能够从数据中学习并改进，而无需明确编程。"},
        {"role": "user", "content": "能举个例子吗？"},
        {"role": "assistant", "content": "当然！比如垃圾邮件过滤器就是机器学习的应用。系统通过学习大量的邮件样本，学会识别哪些是垃圾邮件，哪些是正常邮件。"},
    ]

    print("\n对话内容:")
    for i, msg in enumerate(conversation, 1):
        tokens = counter.count_tokens_openai(msg["content"])
        print(f"\n{i}. [{msg['role']}] ({tokens} tokens)")
        print(f"   {msg['content'][:50]}...")

    # 分析整个对话
    analysis = counter.analyze_conversation(conversation, "gpt-3.5-turbo")

    print("\n" + "-" * 60)
    print(f"总Token数: {analysis['total_tokens']}")
    print(f"消息数量: {analysis['message_count']}")
    print(f"平均每条消息: {analysis['avg_tokens_per_message']:.1f} tokens")

    # 估算成本
    # 假设输入是所有消息，输出是assistant的消息
    input_tokens = analysis['total_tokens']
    output_tokens = sum(
        counter.count_tokens_openai(msg["content"])
        for msg in conversation if msg["role"] == "assistant"
    )

    cost = counter.estimate_cost(input_tokens, output_tokens, "gpt-3.5-turbo")
    print(f"\n估算成本: ${cost:.6f} (¥{cost*7:.4f})")


def demo_real_api_call():
    """演示真实API调用的token统计"""
    print("\n" + "=" * 60)
    print("示例4: 真实API调用Token统计")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  跳过：未配置OPENAI_API_KEY")
        return

    counter = TokenCounter()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = "用一句话解释什么是深度学习"

    print(f"\n提示词: {prompt}")
    print(f"预估输入tokens: {counter.count_tokens_openai(prompt)}")

    # 调用API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )

    answer = response.choices[0].message.content
    actual_input = response.usage.prompt_tokens
    actual_output = response.usage.completion_tokens
    actual_total = response.usage.total_tokens

    print(f"\nAI回复: {answer}")
    print(f"\n实际Token使用:")
    print(f"  输入: {actual_input} tokens")
    print(f"  输出: {actual_output} tokens")
    print(f"  总计: {actual_total} tokens")

    # 计算成本
    cost = counter.estimate_cost(actual_input, actual_output, "gpt-3.5-turbo")
    print(f"\n本次调用成本: ${cost:.6f} (¥{cost*7:.4f})")


def demo_token_limits():
    """演示不同模型的token限制"""
    print("\n" + "=" * 60)
    print("示例5: 模型Token限制")
    print("=" * 60)

    limits = {
        "gpt-4o": {"context": 128000, "output": 16384},
        "gpt-4o-mini": {"context": 128000, "output": 16384},
        "gpt-3.5-turbo": {"context": 16385, "output": 4096},
        "claude-3-5-sonnet-20241022": {"context": 200000, "output": 8192},
        "claude-3-5-haiku-20241022": {"context": 200000, "output": 8192},
        "claude-3-opus-20240229": {"context": 200000, "output": 4096},
    }

    print("\n模型上下文窗口和最大输出:")
    print("-" * 60)
    print(f"{'模型':<35s} {'上下文':<12s} {'最大输出':<12s}")
    print("-" * 60)

    for model, limit in limits.items():
        print(f"{model:<35s} {limit['context']:<12,d} {limit['output']:<12,d}")

    print("\n💡 提示:")
    print("- 上下文窗口 = 输入 + 输出的总token限制")
    print("- 超过限制会导致API调用失败")
    print("- 长对话需要定期清理历史或使用摘要")


def demo_optimization_tips():
    """演示token优化技巧"""
    print("\n" + "=" * 60)
    print("示例6: Token优化技巧")
    print("=" * 60)

    counter = TokenCounter()

    # 技巧1: 精简prompt
    verbose_prompt = """
    我想请你帮我写一个Python函数。这个函数的功能是计算一个列表中所有数字的总和。
    请你用Python语言来实现这个功能。函数应该接收一个列表作为参数，然后返回这个列表中所有元素的和。
    """

    concise_prompt = "写一个Python函数，计算列表中所有数字的总和"

    print("\n技巧1: 精简Prompt")
    print(f"冗长版本: {counter.count_tokens_openai(verbose_prompt)} tokens")
    print(f"精简版本: {counter.count_tokens_openai(concise_prompt)} tokens")
    print(f"节省: {counter.count_tokens_openai(verbose_prompt) - counter.count_tokens_openai(concise_prompt)} tokens")

    # 技巧2: 使用更便宜的模型
    print("\n技巧2: 选择合适的模型")
    print("简单任务用gpt-3.5-turbo或claude-haiku")
    print("复杂任务才用gpt-4或claude-opus")

    input_tokens = 100
    output_tokens = 200

    cheap_cost = counter.estimate_cost(input_tokens, output_tokens, "gpt-3.5-turbo")
    expensive_cost = counter.estimate_cost(input_tokens, output_tokens, "gpt-4o")

    print(f"gpt-3.5-turbo: ${cheap_cost:.6f}")
    print(f"gpt-4o: ${expensive_cost:.6f}")
    print(f"价格差异: {expensive_cost/cheap_cost:.1f}x")

    # 技巧3: 限制max_tokens
    print("\n技巧3: 合理设置max_tokens")
    print("- 不需要长回复时，设置较小的max_tokens")
    print("- 避免不必要的输出token消耗")

    # 技巧4: 清理对话历史
    print("\n技巧4: 管理对话历史")
    print("- 长对话会累积大量tokens")
    print("- 定期清理或只保留最近N轮对话")
    print("- 使用摘要压缩历史信息")


def interactive_calculator():
    """交互式token计算器"""
    print("\n" + "=" * 60)
    print("🧮 交互式Token计算器")
    print("=" * 60)

    counter = TokenCounter()

    print("\n输入文本来计算tokens（输入'quit'退出）:")

    while True:
        text = input("\n> ").strip()

        if text.lower() in ['quit', 'exit', 'q']:
            break

        if not text:
            continue

        tokens = counter.count_tokens_openai(text)
        print(f"\nToken数: {tokens}")
        print(f"字符数: {len(text)}")

        # 估算不同模型的成本（假设这是输入）
        print("\n如果作为输入，成本估算:")
        for model in ["gpt-3.5-turbo", "gpt-4o-mini", "claude-3-5-haiku-20241022"]:
            cost = counter.estimate_cost(tokens, 0, model)
            if cost:
                print(f"  {model}: ${cost:.6f}")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("📊 Token计数和成本估算工具")
    print("=" * 60)

    demo_basic_counting()
    demo_cost_estimation()
    demo_conversation_analysis()
    demo_real_api_call()
    demo_token_limits()
    demo_optimization_tips()

    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成！")
    print("=" * 60)

    print("\n💡 关键要点:")
    print("1. Token ≈ 0.75个英文单词 ≈ 1.5-2个中文字符")
    print("2. 成本 = 输入tokens × 输入价格 + 输出tokens × 输出价格")
    print("3. 不同模型价格差异很大，选择合适的模型")
    print("4. 优化prompt可以显著降低成本")
    print("5. 长对话需要管理历史以控制token使用")

    # 可选：启动交互式计算器
    print("\n是否启动交互式计算器？(y/n): ", end="")
    choice = input().strip().lower()
    if choice == 'y':
        interactive_calculator()


if __name__ == "__main__":
    main()
