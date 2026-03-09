"""
Memory机制演示
展示LangChain中不同类型的Memory使用
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory
)
from langchain.prompts import PromptTemplate

# 加载环境变量
load_dotenv()


def demo_buffer_memory():
    """演示ConversationBufferMemory - 保存所有对话历史"""
    print("=" * 50)
    print("1. ConversationBufferMemory - 完整历史")
    print("=" * 50)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 创建memory
    memory = ConversationBufferMemory()

    # 创建对话链
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    # 多轮对话
    print("\n对话1:")
    response1 = conversation.predict(input="我叫张三，是一名程序员")
    print(f"AI: {response1}")

    print("\n对话2:")
    response2 = conversation.predict(input="我喜欢Python和AI")
    print(f"AI: {response2}")

    print("\n对话3:")
    response3 = conversation.predict(input="你还记得我的名字和职业吗？")
    print(f"AI: {response3}")

    # 查看memory内容
    print("\n" + "-" * 50)
    print("Memory内容:")
    print(memory.load_memory_variables({}))


def demo_window_memory():
    """演示ConversationBufferWindowMemory - 只保留最近K轮对话"""
    print("\n" + "=" * 50)
    print("2. ConversationBufferWindowMemory - 滑动窗口")
    print("=" * 50)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 只保留最近2轮对话
    memory = ConversationBufferWindowMemory(k=2)

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    # 多轮对话
    conversations = [
        "我叫李四",
        "我今年25岁",
        "我住在北京",
        "我是一名设计师",
        "你还记得我的名字吗？"  # 应该不记得，因为超出窗口
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n对话{i}: {user_input}")
        response = conversation.predict(input=user_input)
        print(f"AI: {response}")

    # 查看memory内容
    print("\n" + "-" * 50)
    print("Memory内容（只保留最近2轮）:")
    print(memory.load_memory_variables({}))


def demo_summary_memory():
    """演示ConversationSummaryMemory - 总结历史对话"""
    print("\n" + "=" * 50)
    print("3. ConversationSummaryMemory - 对话摘要")
    print("=" * 50)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 创建摘要memory
    memory = ConversationSummaryMemory(llm=llm)

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    # 多轮对话
    conversations = [
        "我是一名软件工程师，在一家互联网公司工作",
        "我主要负责后端开发，使用Python和Go语言",
        "我的团队有10个人，我们正在开发一个AI项目",
        "根据我之前说的，总结一下我的工作情况"
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n对话{i}: {user_input}")
        response = conversation.predict(input=user_input)
        print(f"AI: {response}")

    # 查看摘要
    print("\n" + "-" * 50)
    print("对话摘要:")
    print(memory.load_memory_variables({}))


def demo_summary_buffer_memory():
    """演示ConversationSummaryBufferMemory - 混合模式"""
    print("\n" + "=" * 50)
    print("4. ConversationSummaryBufferMemory - 混合模式")
    print("=" * 50)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 当token超过100时，旧对话会被总结
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=100
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    # 多轮对话
    conversations = [
        "我正在学习AI Agent开发",
        "我已经学会了LangChain的基础用法",
        "现在我在学习Memory机制",
        "我觉得Memory对于构建聊天机器人很重要",
        "总结一下我在学什么"
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n对话{i}: {user_input}")
        response = conversation.predict(input=user_input)
        print(f"AI: {response}")

    # 查看memory
    print("\n" + "-" * 50)
    print("Memory内容（混合模式）:")
    memory_vars = memory.load_memory_variables({})
    print(memory_vars)


def demo_custom_memory():
    """演示自定义Memory配置"""
    print("\n" + "=" * 50)
    print("5. 自定义Memory配置")
    print("=" * 50)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 自定义memory的key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 自定义prompt
    prompt = PromptTemplate(
        input_variables=["chat_history", "input"],
        template="""你是一个友好的AI助手。

对话历史:
{chat_history}

用户: {input}
AI助手:"""
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )

    # 对话
    print("\n对话1:")
    response1 = conversation.predict(input="你好！")
    print(f"AI: {response1}")

    print("\n对话2:")
    response2 = conversation.predict(input="能帮我写一段Python代码吗？")
    print(f"AI: {response2}")


def demo_memory_operations():
    """演示Memory的基本操作"""
    print("\n" + "=" * 50)
    print("6. Memory基本操作")
    print("=" * 50)

    memory = ConversationBufferMemory()

    # 手动添加对话
    print("\n手动添加对话:")
    memory.save_context(
        {"input": "你好"},
        {"output": "你好！有什么可以帮助你的吗？"}
    )
    memory.save_context(
        {"input": "今天天气怎么样？"},
        {"output": "抱歉，我无法获取实时天气信息。"}
    )

    # 查看memory
    print("\nMemory内容:")
    print(memory.load_memory_variables())

    # 清空memory
    print("\n清空Memory...")
    memory.clear()

    print("\n清空后的Memory:")
    print(memory.load_memory_variables({}))


def demo_memory_comparison():
    """对比不同Memory类型的特点"""
    print("\n" + "=" * 50)
    print("7. Memory类型对比")
    print("=" * 50)

    comparison = """
    Memory类型对比:

    1. ConversationBufferMemory
       - 保存: 所有对话历史
       - 优点: 完整的上下文
       - 缺点: token消耗大，成本高
       - 适用: 短对话，需要完整历史

    2. ConversationBufferWindowMemory
       - 保存: 最近K轮对话
       - 优点: token消耗可控
       - 缺点: 会丢失早期信息
       - 适用: 长对话，只需要近期上下文

    3. ConversationSummaryMemory
       - 保存: 对话摘要
       - 优点: token消耗小，保留关键信息
       - 缺点: 需要额外的摘要调用，可能丢失细节
       - 适用: 长对话，需要保留关键信息

    4. ConversationSummaryBufferMemory
       - 保存: 近期对话 + 早期摘要
       - 优点: 平衡了完整性和效率
       - 缺点: 实现复杂
       - 适用: 长对话，需要平衡性能和上下文

    选择建议:
    - 短对话（<10轮）: BufferMemory
    - 中等对话（10-50轮）: WindowMemory (k=5-10)
    - 长对话（>50轮）: SummaryBufferMemory
    - 需要精确历史: BufferMemory
    - 成本敏感: WindowMemory 或 SummaryMemory
    """

    print(comparison)


def main():
    """主函数"""
    print("\n🧠 LangChain Memory机制演示\n")

    try:
        # 1. Buffer Memory
        demo_buffer_memory()

        # 2. Window Memory
        demo_window_memory()

        # 3. Summary Memory
        demo_summary_memory()

        # 4. Summary Buffer Memory
        demo_summary_buffer_memory()

        # 5. 自定义Memory
        demo_custom_memory()

        # 6. Memory操作
        demo_memory_operations()

        # 7. Memory对比
        demo_memory_comparison()

        print("\n" + "=" * 50)
        print("✅ 所有演示完成！")
        print("=" * 50)

        print("\n💡 关键要点:")
        print("1. Memory让Agent能够记住对话历史")
        print("2. 不同Memory类型适用于不同场景")
        print("3. 需要平衡上下文完整性和token成本")
        print("4. 可以手动操作Memory（保存、清空）")
        print("5. 长对话建议使用Window或Summary模式")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请检查API配置和网络连接")


if __name__ == "__main__":
    main()
