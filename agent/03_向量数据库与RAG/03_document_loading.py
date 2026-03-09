"""
文档加载和分块策略演示
学习如何处理不同格式的文档
"""

import os
from dotenv import load_dotenv
from typing import List, Dict
import time

# 加载环境变量
load_dotenv()


# ============ 文本分块策略 ============

def demo_basic_chunking():
    """演示基础文本分块"""
    print("=" * 60)
    print("1. 基础文本分块")
    print("=" * 60)

    # 示例长文本
    long_text = """
    人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
    它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

    机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下学习。
    机器学习算法通过从数据中学习模式来做出预测或决策。

    深度学习是机器学习的一个子集，它使用多层神经网络来学习数据的表示。
    深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展。

    自然语言处理（NLP）是人工智能的另一个重要领域，它使计算机能够理解、
    解释和生成人类语言。NLP技术被广泛应用于机器翻译、情感分析和问答系统等。
    """

    print("\n原始文本:")
    print("-" * 60)
    print(long_text.strip())
    print(f"\n文本长度: {len(long_text)} 字符")

    # 策略1: 固定大小分块
    print("\n\n策略1: 固定大小分块 (200字符)")
    print("-" * 60)

    chunk_size = 200
    chunks = []
    for i in range(0, len(long_text), chunk_size):
        chunk = long_text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)}字符):")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)

    # 策略2: 按段落分块
    print("\n\n策略2: 按段落分块")
    print("-" * 60)

    paragraphs = [p.strip() for p in long_text.split('\n\n') if p.strip()]

    for i, para in enumerate(paragraphs, 1):
        print(f"\n段落 {i} ({len(para)}字符):")
        print(para[:100] + "..." if len(para) > 100 else para)

    return long_text


def demo_langchain_splitters():
    """演示LangChain的文本分割器"""
    print("\n" + "=" * 60)
    print("2. LangChain文本分割器")
    print("=" * 60)

    try:
        from langchain.text_splitter import (
            CharacterTextSplitter,
            RecursiveCharacterTextSplitter,
            TokenTextSplitter
        )

        # 示例文本
        text = """
        LangChain是一个用于开发由语言模型驱动的应用程序的框架。
        它提供了一系列工具和组件，使开发者能够轻松构建复杂的AI应用。

        LangChain的核心组件包括：
        1. Models - 语言模型的统一接口
        2. Prompts - 提示模板管理
        3. Chains - 将多个组件链接在一起
        4. Agents - 能够使用工具的智能代理
        5. Memory - 对话历史管理

        使用LangChain，开发者可以快速构建聊天机器人、问答系统、
        文档分析工具等各种AI应用。框架提供了丰富的文档和示例，
        帮助开发者快速上手。
        """

        # 1. CharacterTextSplitter
        print("\n1. CharacterTextSplitter (按字符分割)")
        print("-" * 60)

        char_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=200,
            chunk_overlap=20,
            length_function=len
        )

        char_chunks = char_splitter.split_text(text)
        print(f"分块数量: {len(char_chunks)}")

        for i, chunk in enumerate(char_chunks, 1):
            print(f"\nChunk {i} ({len(chunk)}字符):")
            print(chunk[:100] + "..." if len(chunk) > 100 else chunk)

        # 2. RecursiveCharacterTextSplitter
        print("\n\n2. RecursiveCharacterTextSplitter (递归分割)")
        print("-" * 60)

        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )

        recursive_chunks = recursive_splitter.split_text(text)
        print(f"分块数量: {len(recursive_chunks)}")

        for i, chunk in enumerate(recursive_chunks, 1):
            print(f"\nChunk {i} ({len(chunk)}字符):")
            print(chunk[:100] + "..." if len(chunk) > 100 else chunk)

        # 3. TokenTextSplitter
        print("\n\n3. TokenTextSplitter (按Token分割)")
        print("-" * 60)

        token_splitter = TokenTextSplitter(
            chunk_size=100,
            chunk_overlap=10
        )

        token_chunks = token_splitter.split_text(text)
        print(f"分块数量: {len(token_chunks)}")

        for i, chunk in enumerate(token_chunks[:3], 1):  # 只显示前3个
            print(f"\nChunk {i}:")
            print(chunk[:100] + "..." if len(chunk) > 100 else chunk)

    except ImportError:
        print("\n❌ 需要安装langchain: pip install langchain")
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")


def demo_document_loaders():
    """演示文档加载器"""
    print("\n" + "=" * 60)
    print("3. 文档加载器演示")
    print("=" * 60)

    try:
        from langchain.document_loaders import (
            TextLoader,
            DirectoryLoader
        )
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        # 创建示例文本文件
        os.makedirs("data", exist_ok=True)

        sample_texts = [
            ("data/doc1.txt", "这是第一个文档。它包含关于Python编程的内容。\nPython是一种高级编程语言。"),
            ("data/doc2.txt", "这是第二个文档。它讨论机器学习的基础知识。\n机器学习是AI的重要分支。"),
            ("data/doc3.txt", "这是第三个文档。介绍深度学习的应用。\n深度学习在图像识别中很有用。"),
        ]

        print("\n创建示例文档...")
        for filepath, content in sample_texts:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        print(f"✅ 已创建 {len(sample_texts)} 个文档")

        # 1. 加载单个文件
        print("\n\n1. 加载单个文件")
        print("-" * 60)

        loader = TextLoader("data/doc1.txt", encoding='utf-8')
        documents = loader.load()

        print(f"加载的文档数: {len(documents)}")
        for doc in documents:
            print(f"\n内容: {doc.page_content}")
            print(f"元数据: {doc.metadata}")

        # 2. 加载目录
        print("\n\n2. 加载整个目录")
        print("-" * 60)

        dir_loader = DirectoryLoader(
            "data/",
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )

        all_documents = dir_loader.load()
        print(f"加载的文档数: {len(all_documents)}")

        for i, doc in enumerate(all_documents, 1):
            print(f"\n文档 {i}:")
            print(f"  来源: {doc.metadata.get('source', 'unknown')}")
            print(f"  内容: {doc.page_content[:50]}...")

        # 3. 分块处理
        print("\n\n3. 文档分块")
        print("-" * 60)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=10
        )

        chunks = splitter.split_documents(all_documents)
        print(f"分块数量: {len(chunks)}")

        for i, chunk in enumerate(chunks[:5], 1):  # 只显示前5个
            print(f"\nChunk {i}:")
            print(f"  来源: {chunk.metadata.get('source', 'unknown')}")
            print(f"  内容: {chunk.page_content}")

    except ImportError:
        print("\n❌ 需要安装langchain: pip install langchain")
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")


def demo_advanced_chunking():
    """演示高级分块策略"""
    print("\n" + "=" * 60)
    print("4. 高级分块策略")
    print("=" * 60)

    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        # 示例：代码文档
        code_text = """
        # Python函数示例

        def calculate_sum(a, b):
            '''计算两个数的和'''
            return a + b

        def calculate_product(a, b):
            '''计算两个数的积'''
            return a * b

        # 使用示例
        result1 = calculate_sum(5, 3)
        result2 = calculate_product(4, 6)

        print(f"Sum: {result1}")
        print(f"Product: {result2}")
        """

        print("\n示例：代码文档分块")
        print("-" * 60)

        # 针对代码的分割器
        code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=20,
            separators=["\n\n", "\n", " ", ""]
        )

        code_chunks = code_splitter.split_text(code_text)
        print(f"分块数量: {len(code_chunks)}")

        for i, chunk in enumerate(code_chunks, 1):
            print(f"\nChunk {i}:")
            print(chunk)

        # 分块质量评估
        print("\n\n分块质量评估")
        print("-" * 60)

        total_chars = sum(len(chunk) for chunk in code_chunks)
        avg_chunk_size = total_chars / len(code_chunks)

        print(f"总字符数: {total_chars}")
        print(f"平均块大小: {avg_chunk_size:.1f} 字符")
        print(f"最大块大小: {max(len(chunk) for chunk in code_chunks)} 字符")
        print(f"最小块大小: {min(len(chunk) for chunk in code_chunks)} 字符")

    except Exception as e:
        print(f"\n❌ 演示失败: {e}")


def show_chunking_best_practices():
    """显示分块最佳实践"""
    print("\n" + "=" * 60)
    print("5. 分块最佳实践")
    print("=" * 60)

    best_practices = """
╔══════════════════════════════════════════════════════════════╗
║                    分块策略选择指南                          ║
╚══════════════════════════════════════════════════════════════╝

1. Chunk Size（块大小）选择

   文档类型          推荐大小        说明
   ─────────────────────────────────────────────────
   问答对            200-300        短小精悍
   文章段落          500-800        保持语义完整
   技术文档          800-1200       包含完整概念
   书籍章节          1000-1500      保持上下文
   代码文件          300-500        保持函数完整

2. Overlap（重叠）设置

   • 推荐: 10-20% 的chunk_size
   • 作用: 避免信息在边界处丢失
   • 示例: chunk_size=500, overlap=50-100

3. 分隔符选择

   优先级顺序:
   1. 段落分隔符: "\n\n"
   2. 句子分隔符: "。", "！", "？"
   3. 短语分隔符: "，", "；"
   4. 单词分隔符: " "
   5. 字符分隔符: ""

4. 不同内容类型的策略

   📄 普通文本:
      • RecursiveCharacterTextSplitter
      • chunk_size=500, overlap=50

   💻 代码:
      • RecursiveCharacterTextSplitter
      • 保持函数/类完整
      • chunk_size=300, overlap=30

   📊 表格数据:
      • 按行分割
      • 保持表头
      • 添加上下文信息

   📝 Markdown:
      • 按标题层级分割
      • 保持格式
      • MarkdownTextSplitter

╔══════════════════════════════════════════════════════════════╗
║                    常见问题和解决方案                        ║
╚══════════════════════════════════════════════════════════════╝

Q1: Chunk太大会怎样？
A:  • 检索不精确
    • Token成本高
    • 响应变慢
    解决: 减小chunk_size

Q2: Chunk太小会怎样？
A:  • 上下文不完整
    • 需要检索更多chunks
    • 可能丢失关键信息
    解决: 增大chunk_size或overlap

Q3: 如何处理长表格？
A:  • 按行分割
    • 每个chunk包含表头
    • 添加表格标题作为metadata

Q4: 如何保持代码完整性？
A:  • 使用代码感知的分割器
    • 按函数/类分割
    • 保持缩进结构

Q5: 多语言文档如何处理？
A:  • 使用支持多语言的分割器
    • 注意不同语言的分隔符
    • 考虑使用TokenTextSplitter

╔══════════════════════════════════════════════════════════════╗
║                    性能优化建议                              ║
╚══════════════════════════════════════════════════════════════╝

1. 批量处理
   • 一次处理多个文档
   • 使用并行处理

2. 缓存结果
   • 缓存分块结果
   • 避免重复处理

3. 增量更新
   • 只处理新增/修改的文档
   • 保持索引更新

4. 内存管理
   • 流式处理大文件
   • 及时释放内存

╔══════════════════════════════════════════════════════════════╗
║                    实践建议                                  ║
╚══════════════════════════════════════════════════════════════╝

1. 从默认配置开始
   chunk_size=500
   chunk_overlap=50

2. 根据实际效果调整
   • 检索质量不好 → 调整chunk_size
   • 上下文不完整 → 增加overlap
   • 成本太高 → 减小chunk_size

3. A/B测试
   • 测试不同配置
   • 评估检索质量
   • 选择最佳参数

4. 监控和优化
   • 记录分块统计
   • 分析检索效果
   • 持续优化
"""

    print(best_practices)


def main():
    """主函数"""
    print("\n📄 文档加载和分块策略演示\n")

    try:
        # 1. 基础分块
        demo_basic_chunking()

        # 2. LangChain分割器
        demo_langchain_splitters()

        # 3. 文档加载器
        demo_document_loaders()

        # 4. 高级分块
        demo_advanced_chunking()

        # 5. 最佳实践
        show_chunking_best_practices()

        print("\n" + "=" * 60)
        print("✅ 所有演示完成！")
        print("=" * 60)

        print("\n💡 关键要点:")
        print("1. 选择合适的chunk_size很重要")
        print("2. overlap可以避免信息丢失")
        print("3. 不同内容类型需要不同策略")
        print("4. RecursiveCharacterTextSplitter最常用")
        print("5. 需要根据实际效果调整参数")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请检查:")
        print("1. 是否安装了langchain")
        print("2. data目录是否存在")


if __name__ == "__main__":
    main()
