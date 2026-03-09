"""
LangChain基础组件示例
演示Models, Prompts, Chains的基本使用
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.schema import HumanMessage, SystemMessage

# 加载环境变量
load_dotenv()


def demo_models():
    """演示不同的模型使用"""
    print("=" * 50)
    print("1. Models演示")
    print("=" * 50)

    # OpenAI模型
    print("\n[OpenAI GPT-4o-mini]")
    openai_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=100
    )

    response = openai_llm.invoke("用一句话解释什么是Agent")
    print(f"回答: {response.content}")

    # Anthropic模型
    print("\n[Anthropic Claude-3-Haiku]")
    anthropic_llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0.7,
        max_tokens=100
    )

    response = anthropic_llm.invoke("用一句话解释什么是Agent")
    print(f"回答: {response.content}")


def demo_prompts():
    """演示Prompt模板的使用"""
    print("\n" + "=" * 50)
    print("2. Prompts演示")
    print("=" * 50)

    # 简单的PromptTemplate
    print("\n[简单Prompt模板]")
    simple_template = PromptTemplate(
        input_variables=["topic"],
        template="请用3句话介绍{topic}。"
    )

    prompt = simple_template.format(topic="LangChain")
    print(f"生成的Prompt: {prompt}")

    # ChatPromptTemplate
    print("\n[Chat Prompt模板]")
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，擅长{skill}。"),
        ("human", "{question}")
    ])

    messages = chat_template.format_messages(
        role="AI助手",
        skill="解释复杂概念",
        question="什么是向量数据库？"
    )

    print("生成的消息:")
    for msg in messages:
        print(f"  {msg.type}: {msg.content}")

    # Few-shot示例
    print("\n[Few-shot Prompt]")
    few_shot_template = PromptTemplate(
        input_variables=["input"],
        template="""请将以下句子翻译成英文。

示例1:
输入: 你好，世界
输出: Hello, World

示例2:
输入: 今天天气真好
输出: The weather is really nice today

现在翻译:
输入: {input}
输出:"""
    )

    prompt = few_shot_template.format(input="我喜欢学习AI")
    print(f"生成的Prompt:\n{prompt}")


def demo_chains():
    """演示Chain的使用"""
    print("\n" + "=" * 50)
    print("3. Chains演示")
    print("=" * 50)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # LLMChain - 基础链
    print("\n[LLMChain - 基础链]")
    prompt = PromptTemplate(
        input_variables=["product"],
        template="为{product}写一句广告语。"
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"product": "智能手表"})
    print(f"结果: {result['text']}")

    # SequentialChain - 顺序链
    print("\n[SequentialChain - 顺序链]")

    # 第一个链：生成故事大纲
    outline_prompt = PromptTemplate(
        input_variables=["topic"],
        template="为一个关于{topic}的短故事写一个大纲（3-4句话）。"
    )
    outline_chain = LLMChain(
        llm=llm,
        prompt=outline_prompt,
        output_key="outline"
    )

    # 第二个链：根据大纲写故事
    story_prompt = PromptTemplate(
        input_variables=["outline"],
        template="根据以下大纲，写一个100字左右的短故事：\n\n{outline}"
    )
    story_chain = LLMChain(
        llm=llm,
        prompt=story_prompt,
        output_key="story"
    )

    # 组合成顺序链
    overall_chain = SequentialChain(
        chains=[outline_chain, story_chain],
        input_variables=["topic"],
        output_variables=["outline", "story"],
        verbose=True
    )

    result = overall_chain.invoke({"topic": "AI机器人"})
    print(f"\n大纲:\n{result['outline']}")
    print(f"\n故事:\n{result['story']}")


def demo_chain_with_multiple_inputs():
    """演示多输入的Chain"""
    print("\n" + "=" * 50)
    print("4. 多输入Chain演示")
    print("=" * 50)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 创建一个需要多个输入的prompt
    prompt = PromptTemplate(
        input_variables=["style", "topic", "length"],
        template="""请用{style}的风格写一篇关于{topic}的文章。
文章长度：{length}字左右。"""
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    result = chain.invoke({
        "style": "幽默诙谐",
        "topic": "程序员的日常",
        "length": "150"
    })

    print(f"结果:\n{result['text']}")


def demo_streaming():
    """演示流式输出"""
    print("\n" + "=" * 50)
    print("5. 流式输出演示")
    print("=" * 50)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        streaming=True
    )

    print("\n正在生成回答（流式输出）:")
    print("-" * 50)

    for chunk in llm.stream("用50字介绍Python编程语言的特点"):
        print(chunk.content, end="", flush=True)

    print("\n" + "-" * 50)


def main():
    """主函数"""
    print("\n🚀 LangChain基础组件演示\n")

    try:
        # 1. Models演示
        demo_models()

        # 2. Prompts演示
        demo_prompts()

        # 3. Chains演示
        demo_chains()

        # 4. 多输入Chain
        demo_chain_with_multiple_inputs()

        # 5. 流式输出
        demo_streaming()

        print("\n" + "=" * 50)
        print("✅ 所有演示完成！")
        print("=" * 50)

        print("\n💡 关键要点:")
        print("1. Models: LangChain提供统一的模型接口")
        print("2. Prompts: 使用模板管理prompt，提高复用性")
        print("3. Chains: 组合多个步骤，构建复杂流程")
        print("4. 流式输出: 提供更好的用户体验")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请检查:")
        print("1. 是否正确配置了.env文件")
        print("2. API密钥是否有效")
        print("3. 网络连接是否正常")


if __name__ == "__main__":
    main()
