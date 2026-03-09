"""
Embedding基础演示
学习向量表示和相似度计算
"""

import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Tuple
import time

# 加载环境变量
load_dotenv()


# ============ Embedding基础 ============

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    获取文本的Embedding向量

    Args:
        text: 输入文本
        model: Embedding模型

    Returns:
        向量列表
    """
    client = OpenAI()

    # 清理文本
    text = text.replace("\n", " ").strip()

    # 获取embedding
    response = client.embeddings.create(
        input=text,
        model=model
    )

    return response.data[0].embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算余弦相似度

    Args:
        vec1: 向量1
        vec2: 向量2

    Returns:
        相似度分数 (0-1)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # 余弦相似度公式
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    计算欧氏距离

    Args:
        vec1: 向量1
        vec2: 向量2

    Returns:
        距离值（越小越相似）
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return np.linalg.norm(vec1 - vec2)


# ============ 演示函数 ============

def demo_basic_embedding():
    """演示基础Embedding"""
    print("=" * 60)
    print("1. 基础Embedding演示")
    print("=" * 60)

    # 示例文本
    texts = [
        "人工智能是计算机科学的一个分支",
        "AI是一种让机器模拟人类智能的技术",
        "今天天气真好，适合出去玩",
    ]

    print("\n获取文本的Embedding向量...")
    embeddings = []

    for i, text in enumerate(texts, 1):
        print(f"\n文本{i}: {text}")
        embedding = get_embedding(text)
        embeddings.append(embedding)

        print(f"  向量维度: {len(embedding)}")
        print(f"  前5个值: {embedding[:5]}")

    return texts, embeddings


def demo_similarity_calculation(texts: List[str], embeddings: List[List[float]]):
    """演示相似度计算"""
    print("\n" + "=" * 60)
    print("2. 相似度计算演示")
    print("=" * 60)

    print("\n计算文本之间的相似度...")
    print("-" * 60)

    # 计算所有文本对之间的相似度
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            cos_sim = cosine_similarity(embeddings[i], embeddings[j])
            euc_dist = euclidean_distance(embeddings[i], embeddings[j])

            print(f"\n文本{i+1} vs 文本{j+1}:")
            print(f"  文本{i+1}: {texts[i]}")
            print(f"  文本{j+1}: {texts[j]}")
            print(f"  余弦相似度: {cos_sim:.4f}")
            print(f"  欧氏距离: {euc_dist:.4f}")


def demo_semantic_search():
    """演示语义搜索"""
    print("\n" + "=" * 60)
    print("3. 语义搜索演示")
    print("=" * 60)

    # 文档库
    documents = [
        "Python是一种高级编程语言，广泛用于数据科学和AI开发",
        "机器学习是人工智能的一个子领域",
        "深度学习使用神经网络来学习数据中的模式",
        "自然语言处理让计算机能够理解人类语言",
        "计算机视觉使机器能够理解和分析图像",
        "今天的午餐很美味，我吃了意大利面",
        "周末我打算去爬山，天气预报说会很晴朗",
    ]

    print("\n构建文档库...")
    print(f"文档数量: {len(documents)}")

    # 获取所有文档的embedding
    print("\n正在生成文档向量...")
    doc_embeddings = []
    for i, doc in enumerate(documents, 1):
        print(f"  处理文档 {i}/{len(documents)}")
        embedding = get_embedding(doc)
        doc_embeddings.append(embedding)
        time.sleep(0.5)  # 避免API限流

    # 用户查询
    queries = [
        "什么是深度学习？",
        "编程语言有哪些？",
        "周末活动推荐",
    ]

    print("\n" + "=" * 60)
    print("开始语义搜索...")
    print("=" * 60)

    for query in queries:
        print(f"\n查询: {query}")
        print("-" * 60)

        # 获取查询的embedding
        query_embedding = get_embedding(query)

        # 计算与所有文档的相似度
        similarities = []
        for doc_emb in doc_embeddings:
            sim = cosine_similarity(query_embedding, doc_emb)
            similarities.append(sim)

        # 排序并获取Top-3
        top_indices = np.argsort(similarities)[::-1][:3]

        print("\n最相关的文档:")
        for rank, idx in enumerate(top_indices, 1):
            print(f"\n  {rank}. 相似度: {similarities[idx]:.4f}")
            print(f"     文档: {documents[idx]}")


def demo_embedding_properties():
    """演示Embedding的特性"""
    print("\n" + "=" * 60)
    print("4. Embedding特性演示")
    print("=" * 60)

    # 测试不同语言
    print("\n测试1: 多语言相似性")
    print("-" * 60)

    texts = [
        "Hello, how are you?",
        "你好，你好吗？",
        "Bonjour, comment allez-vous?",
    ]

    embeddings = [get_embedding(text) for text in texts]

    print("\n英文 vs 中文:")
    sim = cosine_similarity(embeddings[0], embeddings[1])
    print(f"  相似度: {sim:.4f}")

    print("\n英文 vs 法文:")
    sim = cosine_similarity(embeddings[0], embeddings[2])
    print(f"  相似度: {sim:.4f}")

    # 测试语义相似性
    print("\n\n测试2: 语义相似性")
    print("-" * 60)

    pairs = [
        ("国王", "王后"),
        ("男人", "女人"),
        ("大", "小"),
        ("快乐", "悲伤"),
    ]

    for word1, word2 in pairs:
        emb1 = get_embedding(word1)
        emb2 = get_embedding(word2)
        sim = cosine_similarity(emb1, emb2)
        print(f"\n{word1} vs {word2}: {sim:.4f}")
        time.sleep(0.5)


def demo_embedding_dimensions():
    """演示Embedding维度信息"""
    print("\n" + "=" * 60)
    print("5. Embedding维度分析")
    print("=" * 60)

    text = "这是一个测试文本"
    embedding = get_embedding(text)

    embedding_array = np.array(embedding)

    print(f"\n文本: {text}")
    print(f"向量维度: {len(embedding)}")
    print(f"向量范数: {np.linalg.norm(embedding_array):.4f}")
    print(f"最大值: {np.max(embedding_array):.4f}")
    print(f"最小值: {np.min(embedding_array):.4f}")
    print(f"平均值: {np.mean(embedding_array):.4f}")
    print(f"标准差: {np.std(embedding_array):.4f}")


def main():
    """主函数"""
    print("\n🎯 Embedding基础演示\n")

    try:
        # 1. 基础Embedding
        texts, embeddings = demo_basic_embedding()

        # 2. 相似度计算
        demo_similarity_calculation(texts, embeddings)

        # 3. 语义搜索
        demo_semantic_search()

        # 4. Embedding特性
        demo_embedding_properties()

        # 5. 维度分析
        demo_embedding_dimensions()

        print("\n" + "=" * 60)
        print("✅ 所有演示完成！")
        print("=" * 60)

        print("\n💡 关键要点:")
        print("1. Embedding将文本转换为向量表示")
        print("2. 相似的文本在向量空间中距离更近")
        print("3. 余弦相似度是最常用的相似度度量")
        print("4. OpenAI ada-002模型生成1536维向量")
        print("5. Embedding支持多语言和语义理解")

        print("\n📊 性能指标:")
        print("• 向量维度: 1536")
        print("• 单次调用: ~0.5秒")
        print("• 成本: $0.0001/1K tokens")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请检查:")
        print("1. API密钥配置")
        print("2. 网络连接")
        print("3. API额度")


if __name__ == "__main__":
    main()
