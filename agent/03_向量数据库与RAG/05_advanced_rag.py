"""
高级RAG技巧演示
包括重排序、混合搜索、Query改写等
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import time

# 加载环境变量
load_dotenv()


# ============ Query改写 ============

def demo_query_rewriting():
    """演示Query改写技巧"""
    print("=" * 60)
    print("1. Query改写 (Query Rewriting)")
    print("=" * 60)

    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import PromptTemplate

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 原始查询
        original_query = "深度学习好用吗"

        print(f"\n原始查询: {original_query}")
        print("-" * 60)

        # Query改写Prompt
        rewrite_template = """你是一个查询优化专家。请将用户的查询改写为更适合检索的形式。

要求:
1. 使用更专业的术语
2. 补充相关关键词
3. 保持查询意图不变
4. 只返回改写后的查询，不要解释

原始查询: {query}

改写后的查询:"""

        prompt = PromptTemplate(
            template=rewrite_template,
            input_variables=["query"]
        )

        # 执行改写
        from langchain.schema import HumanMessage
        rewrite_prompt = prompt.format(query=original_query)
        response = llm.invoke([HumanMessage(content=rewrite_prompt)])

        rewritten_query = response.content.strip()

        print(f"\n改写后查询: {rewritten_query}")

        # 多查询生成
        print("\n\n多查询生成 (Multi-Query)")
        print("-" * 60)

        multi_query_template = """基于用户的问题，生成3个不同角度的相关查询。

要求:
1. 每个查询从不同角度理解问题
2. 使用不同的关键词
3. 每行一个查询

原始问题: {query}

相关查询:"""

        multi_prompt = PromptTemplate(
            template=multi_query_template,
            input_variables=["query"]
        )

        multi_query_prompt = multi_prompt.format(query=original_query)
        response = llm.invoke([HumanMessage(content=multi_query_prompt)])

        queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]

        print(f"\n生成的查询:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")

        return rewritten_query, queries

    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        return None, None


# ============ HyDE (假设文档嵌入) ============

def demo_hyde():
    """演示HyDE技术"""
    print("\n" + "=" * 60)
    print("2. HyDE (Hypothetical Document Embeddings)")
    print("=" * 60)

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain.prompts import PromptTemplate

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        embeddings = OpenAIEmbeddings()

        # 用户问题
        question = "什么是深度学习？"

        print(f"\n用户问题: {question}")
        print("-" * 60)

        # 生成假设文档
        hyde_template = """请写一段文字来回答以下问题。
这段文字将用于检索相关文档，所以要包含相关的关键词和概念。

问题: {question}

回答:"""

        prompt = PromptTemplate(
            template=hyde_template,
            input_variables=["question"]
        )

        from langchain.schema import HumanMessage
        hyde_prompt = prompt.format(question=question)
        response = llm.invoke([HumanMessage(content=hyde_prompt)])

        hypothetical_doc = response.content.strip()

        print(f"\n生成的假设文档:")
        print(hypothetical_doc)

        # 对比embedding
        print("\n\n对比embedding相似度:")
        print("-" * 60)

        # 问题embedding
        question_emb = embeddings.embed_query(question)

        # 假设文档embedding
        hyde_emb = embeddings.embed_query(hypothetical_doc)

        # 示例目标文档
        target_doc = "深度学习是机器学习的一个子集，使用多层神经网络来学习数据的复杂表示。"
        target_emb = embeddings.embed_query(target_doc)

        # 计算相似度
        import numpy as np

        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        sim_question = cosine_similarity(question_emb, target_emb)
        sim_hyde = cosine_similarity(hyde_emb, target_emb)

        print(f"\n目标文档: {target_doc}")
        print(f"\n直接查询相似度: {sim_question:.4f}")
        print(f"HyDE查询相似度: {sim_hyde:.4f}")
        print(f"提升: {(sim_hyde - sim_question):.4f}")

    except Exception as e:
        print(f"\n❌ 演示失败: {e}")


# ============ 重排序 (Reranking) ============

def demo_reranking():
    """演示重排序技术"""
    print("\n" + "=" * 60)
    print("3. 重排序 (Reranking)")
    print("=" * 60)

    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain.vectorstores import Chroma

        # 加载向量存储
        embeddings = OpenAIEmbeddings()

        if not os.path.exists("chroma_db"):
            print("⚠️  请先运行 04_basic_rag.py 创建向量数据库")
            return

        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        question = "深度学习有哪些应用？"

        print(f"\n问题: {question}")
        print("-" * 60)

        # 第一阶段：向量检索
        print("\n第一阶段：向量检索 (检索更多候选)")
        print("-" * 60)

        # 检索更多文档
        docs_with_scores = vectorstore.similarity_search_with_score(question, k=5)

        print(f"检索到 {len(docs_with_scores)} 个候选文档:")
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            print(f"\n{i}. 向量距离: {score:.4f}")
            print(f"   内容: {doc.page_content[:80]}...")

        # 第二阶段：重排序
        print("\n\n第二阶段：重排序 (使用LLM评分)")
        print("-" * 60)

        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 为每个文档评分
        reranked_docs = []

        for doc, vec_score in docs_with_scores:
            # 评分prompt
            score_prompt = f"""评估以下文档对问题的相关性，给出0-10的分数。
只返回数字，不要解释。

问题: {question}

文档: {doc.page_content}

相关性分数:"""

            response = llm.invoke([HumanMessage(content=score_prompt)])

            try:
                relevance_score = float(response.content.strip())
            except:
                relevance_score = 5.0

            reranked_docs.append((doc, vec_score, relevance_score))

        # 按重排序分数排序
        reranked_docs.sort(key=lambda x: x[2], reverse=True)

        print(f"\n重排序后的文档 (Top-3):")
        for i, (doc, vec_score, rel_score) in enumerate(reranked_docs[:3], 1):
            print(f"\n{i}. 相关性分数: {rel_score:.1f}/10")
            print(f"   向量距离: {vec_score:.4f}")
            print(f"   内容: {doc.page_content[:80]}...")

        print("\n💡 重排序优势:")
        print("  • 提高检索精度")
        print("  • 考虑语义相关性")
        print("  • 过滤低质量结果")

    except Exception as e:
        print(f"\n❌ 演示失败: {e}")


# ============ 混合搜索 ============

def demo_hybrid_search():
    """演示混合搜索"""
    print("\n" + "=" * 60)
    print("4. 混合搜索 (Hybrid Search)")
    print("=" * 60)

    print("\n混合搜索 = 向量搜索 + 关键词搜索")
    print("-" * 60)

    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain.vectorstores import Chroma

        embeddings = OpenAIEmbeddings()

        if not os.path.exists("chroma_db"):
            print("⚠️  请先运行 04_basic_rag.py 创建向量数据库")
            return

        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        question = "机器学习的方法"

        # 1. 向量搜索
        print("\n1. 纯向量搜索:")
        print("-" * 60)

        vector_docs = vectorstore.similarity_search(question, k=3)

        for i, doc in enumerate(vector_docs, 1):
            print(f"\n{i}. {doc.page_content[:80]}...")

        # 2. 关键词过滤
        print("\n\n2. 向量搜索 + 关键词过滤:")
        print("-" * 60)

        # 提取关键词
        keywords = ["机器学习", "方法", "算法"]

        filtered_docs = []
        for doc in vector_docs:
            # 检查是否包含关键词
            if any(kw in doc.page_content for kw in keywords):
                filtered_docs.append(doc)

        print(f"过滤后文档数: {len(filtered_docs)}")
        for i, doc in enumerate(filtered_docs, 1):
            print(f"\n{i}. {doc.page_content[:80]}...")

        # 3. 使用Metadata过滤
        print("\n\n3. 使用Metadata过滤:")
        print("-" * 60)

        # 注意：需要在创建向量存储时添加metadata
        print("示例：只检索特定类别的文档")
        print("vectorstore.similarity_search(")
        print("    question,")
        print("    k=3,")
        print("    filter={'category': 'AI'}")
        print(")")

        print("\n💡 混合搜索优势:")
        print("  • 结合语义和关键词")
        print("  • 提高精确度")
        print("  • 支持复杂过滤")

    except Exception as e:
        print(f"\n❌ 演示失败: {e}")


# ============ 上下文压缩 ============

def demo_context_compression():
    """演示上下文压缩"""
    print("\n" + "=" * 60)
    print("5. 上下文压缩 (Context Compression)")
    print("=" * 60)

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain.vectorstores import Chroma
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor

        embeddings = OpenAIEmbeddings()

        if not os.path.exists("chroma_db"):
            print("⚠️  请先运行 04_basic_rag.py 创建向量数据库")
            return

        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        question = "深度学习的应用"

        # 1. 普通检索
        print("\n1. 普通检索:")
        print("-" * 60)

        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = base_retriever.get_relevant_documents(question)

        print(f"检索到 {len(docs)} 个文档")
        total_chars = sum(len(doc.page_content) for doc in docs)
        print(f"总字符数: {total_chars}")

        for i, doc in enumerate(docs, 1):
            print(f"\n文档 {i} ({len(doc.page_content)}字符):")
            print(doc.page_content[:100] + "...")

        # 2. 压缩检索
        print("\n\n2. 压缩检索:")
        print("-" * 60)

        # 创建压缩器
        compressor = LLMChainExtractor.from_llm(llm)

        # 创建压缩检索器
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        # 执行压缩检索
        compressed_docs = compression_retriever.get_relevant_documents(question)

        print(f"压缩后文档数: {len(compressed_docs)}")
        compressed_chars = sum(len(doc.page_content) for doc in compressed_docs)
        print(f"总字符数: {compressed_chars}")
        print(f"压缩率: {(1 - compressed_chars/total_chars)*100:.1f}%")

        for i, doc in enumerate(compressed_docs, 1):
            print(f"\n文档 {i} ({len(doc.page_content)}字符):")
            print(doc.page_content)

        print("\n💡 上下文压缩优势:")
        print("  • 减少token消耗")
        print("  • 提取关键信息")
        print("  • 提高响应速度")
        print("  • 降低成本")

    except Exception as e:
        print(f"\n❌ 演示失败: {e}")


# ============ Self-Query ============

def demo_self_query():
    """演示Self-Query"""
    print("\n" + "=" * 60)
    print("6. Self-Query (自查询)")
    print("=" * 60)

    print("\nSelf-Query: 自动从查询中提取过滤条件")
    print("-" * 60)

    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import PromptTemplate
        from langchain.schema import HumanMessage

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 用户查询
        user_query = "找一些关于深度学习的初级教程"

        print(f"\n用户查询: {user_query}")

        # 提取过滤条件
        extract_template = """从用户查询中提取搜索条件和过滤条件。

用户查询: {query}

请以JSON格式返回:
{{
    "search_query": "实际搜索的内容",
    "filters": {{
        "category": "类别",
        "level": "难度级别"
    }}
}}

只返回JSON，不要其他内容:"""

        prompt = PromptTemplate(
            template=extract_template,
            input_variables=["query"]
        )

        extract_prompt = prompt.format(query=user_query)
        response = llm.invoke([HumanMessage(content=extract_prompt)])

        print(f"\n提取的条件:")
        print(response.content)

        print("\n💡 Self-Query优势:")
        print("  • 自动提取过滤条件")
        print("  • 提高检索精度")
        print("  • 支持复杂查询")
        print("  • 用户体验好")

    except Exception as e:
        print(f"\n❌ 演示失败: {e}")


# ============ 最佳实践总结 ============

def show_advanced_rag_best_practices():
    """显示高级RAG最佳实践"""
    print("\n" + "=" * 60)
    print("7. 高级RAG最佳实践")
    print("=" * 60)

    best_practices = """
╔══════════════════════════════════════════════════════════════╗
║                    高级RAG技巧对比                           ║
╚══════════════════════════════════════════════════════════════╝

技巧              适用场景              提升效果    成本
─────────────────────────────────────────────────────────
Query改写        查询不精确            ⭐⭐⭐      低
Multi-Query      需要多角度检索        ⭐⭐⭐⭐    中
HyDE             查询与文档差异大      ⭐⭐⭐⭐    中
重排序           需要高精度            ⭐⭐⭐⭐⭐  高
混合搜索         需要精确匹配          ⭐⭐⭐⭐    低
上下文压缩       Token成本高           ⭐⭐⭐⭐    中
Self-Query       复杂过滤需求          ⭐⭐⭐      低

╔══════════════════════════════════════════════════════════════╗
║                    技巧组合建议                              ║
╚══════════════════════════════════════════════════════════════╝

基础RAG:
  向量检索 → LLM生成

进阶RAG:
  Query改写 → 向量检索 → 重排序 → LLM生成

高级RAG:
  Multi-Query → 向量检索 → 混合搜索 → 重排序 →
  上下文压缩 → LLM生成

企业级RAG:
  Self-Query → HyDE → 向量检索 → 混合搜索 →
  重排序 → 上下文压缩 → LLM生成 → 答案验证

╔══════════════════════════════════════════════════════════════╗
║                    实施建议                                  ║
╚══════════════════════════════════════════════════════════════╝

1. 渐进式优化
   • 从基础RAG开始
   • 根据问题逐步添加技巧
   • 不要过度优化

2. 成本效益分析
   • 评估每个技巧的收益
   • 考虑实施成本
   • 选择性使用

3. A/B测试
   • 对比不同配置
   • 收集用户反馈
   • 数据驱动决策

4. 监控和迭代
   • 监控关键指标
   • 持续优化
   • 定期评估

╔══════════════════════════════════════════════════════════════╗
║                    性能优化                                  ║
╚══════════════════════════════════════════════════════════════╝

1. 缓存策略
   • 缓存Query改写结果
   • 缓存检索结果
   • 缓存LLM响应

2. 批量处理
   • 批量生成embedding
   • 批量重排序
   • 并行处理

3. 异步处理
   • 异步检索
   • 异步LLM调用
   • 流式响应

4. 资源优化
   • 使用GPU加速
   • 优化索引结构
   • 减少不必要的调用

╔══════════════════════════════════════════════════════════════╗
║                    常见问题                                  ║
╚══════════════════════════════════════════════════════════════╝

Q1: 应该使用哪些技巧？
A:  根据实际需求选择：
    • 检索不准 → Query改写 + 重排序
    • 成本高 → 上下文压缩
    • 需要精确匹配 → 混合搜索
    • 复杂查询 → Self-Query

Q2: 如何平衡性能和成本？
A:  • 优先使用低成本技巧
    • 关键场景使用高成本技巧
    • 使用缓存降低成本
    • 监控ROI

Q3: 重排序值得吗？
A:  • 高精度需求：值得
    • 成本敏感：慎用
    • 可以先用简单方法
    • 根据效果决定

Q4: 如何评估效果？
A:  • 建立测试集
    • 定义评估指标
    • A/B测试
    • 用户反馈
"""

    print(best_practices)


def main():
    """主函数"""
    print("\n🚀 高级RAG技巧演示\n")

    try:
        # 1. Query改写
        demo_query_rewriting()

        # 2. HyDE
        demo_hyde()

        # 3. 重排序
        demo_reranking()

        # 4. 混合搜索
        demo_hybrid_search()

        # 5. 上下文压缩
        demo_context_compression()

        # 6. Self-Query
        demo_self_query()

        # 7. 最佳实践
        show_advanced_rag_best_practices()

        print("\n" + "=" * 60)
        print("✅ 所有演示完成！")
        print("=" * 60)

        print("\n💡 关键要点:")
        print("1. Query改写可以提高检索质量")
        print("2. 重排序能显著提升精度")
        print("3. 混合搜索结合语义和关键词")
        print("4. 上下文压缩降低成本")
        print("5. 根据需求选择合适的技巧")

        print("\n📊 技巧效果:")
        print("• Query改写: +10-20% 准确率")
        print("• 重排序: +15-30% 准确率")
        print("• 上下文压缩: -30-50% Token")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请检查:")
        print("1. API密钥配置")
        print("2. 向量数据库存在")
        print("3. 依赖包安装")


if __name__ == "__main__":
    main()
