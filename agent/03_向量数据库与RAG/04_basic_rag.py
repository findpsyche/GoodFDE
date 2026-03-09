"""
基础RAG系统实现
构建一个完整的文档问答系统
"""

import os
from dotenv import load_dotenv
from typing import List, Dict
import time

# 加载环境变量
load_dotenv()


# ============ 基础RAG系统 ============

def build_basic_rag():
    """构建基础RAG系统"""
    print("=" * 60)
    print("1. 构建基础RAG系统")
    print("=" * 60)

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.vectorstores import Chroma
        from langchain.chains import RetrievalQA
        from langchain.document_loaders import TextLoader

        # 1. 加载文档
        print("\n步骤1: 加载文档")
        print("-" * 60)

        # 确保示例文档存在
        if not os.path.exists("data/sample.txt"):
            print("创建示例文档...")
            os.makedirs("data", exist_ok=True)
            with open("data/sample.txt", "w", encoding="utf-8") as f:
                f.write("""
人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。

机器学习是AI的核心技术之一，它使计算机能够从数据中学习，而无需明确编程。
主要的机器学习方法包括监督学习、无监督学习和强化学习。

深度学习是机器学习的一个子集，使用多层神经网络来学习数据的复杂表示。
深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展。

自然语言处理（NLP）使计算机能够理解、解释和生成人类语言。
NLP的应用包括机器翻译、情感分析、问答系统和文本摘要。

计算机视觉使机器能够从图像和视频中提取信息和理解内容。
应用包括人脸识别、物体检测、图像分类和自动驾驶。
                """)

        loader = TextLoader("data/sample.txt", encoding="utf-8")
        documents = loader.load()
        print(f"✅ 加载了 {len(documents)} 个文档")

        # 2. 文档分块
        print("\n步骤2: 文档分块")
        print("-" * 60)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=len
        )

        chunks = text_splitter.split_documents(documents)
        print(f"✅ 分成 {len(chunks)} 个块")

        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\nChunk {i}: {chunk.page_content[:80]}...")

        # 3. 创建向量存储
        print("\n步骤3: 创建向量存储")
        print("-" * 60)

        embeddings = OpenAIEmbeddings()

        # 删除旧的数据库
        if os.path.exists("chroma_db"):
            import shutil
            shutil.rmtree("chroma_db")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

        print(f"✅ 向量存储已创建，包含 {len(chunks)} 个向量")

        # 4. 创建检索器
        print("\n步骤4: 创建检索器")
        print("-" * 60)

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        print("✅ 检索器已创建")

        # 5. 创建QA链
        print("\n步骤5: 创建QA链")
        print("-" * 60)

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )

        print("✅ QA链已创建")

        # 6. 测试问答
        print("\n步骤6: 测试问答")
        print("=" * 60)

        questions = [
            "什么是机器学习？",
            "深度学习有哪些应用？",
            "NLP是什么？",
        ]

        for i, question in enumerate(questions, 1):
            print(f"\n问题 {i}: {question}")
            print("-" * 60)

            result = qa_chain.invoke({"query": question})

            print(f"\n答案: {result['result']}")

            print("\n相关文档:")
            for j, doc in enumerate(result['source_documents'], 1):
                print(f"  {j}. {doc.page_content[:100]}...")

        return qa_chain, vectorstore

    except ImportError as e:
        print(f"\n❌ 缺少依赖: {e}")
        print("请安装: pip install langchain langchain-openai chromadb")
        return None, None
    except Exception as e:
        print(f"\n❌ 构建失败: {e}")
        return None, None


# ============ RAG工作流程演示 ============

def demo_rag_workflow():
    """演示RAG的完整工作流程"""
    print("\n" + "=" * 60)
    print("2. RAG工作流程详解")
    print("=" * 60)

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain.vectorstores import Chroma
        from langchain.prompts import PromptTemplate

        # 加载已有的向量存储
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 用户问题
        question = "深度学习在哪些领域有应用？"

        print(f"\n用户问题: {question}")
        print("=" * 60)

        # 步骤1: 问题向量化
        print("\n步骤1: 问题向量化")
        print("-" * 60)
        query_embedding = embeddings.embed_query(question)
        print(f"✅ 问题已转换为 {len(query_embedding)} 维向量")
        print(f"向量前5个值: {query_embedding[:5]}")

        # 步骤2: 向量检索
        print("\n步骤2: 向量检索")
        print("-" * 60)
        docs = vectorstore.similarity_search(question, k=3)
        print(f"✅ 检索到 {len(docs)} 个相关文档")

        for i, doc in enumerate(docs, 1):
            print(f"\n文档 {i}:")
            print(f"  内容: {doc.page_content}")

        # 步骤3: 构建Prompt
        print("\n步骤3: 构建Prompt")
        print("-" * 60)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt_template = """基于以下上下文回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文:
{context}

问题: {question}

答案:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        final_prompt = prompt.format(context=context, question=question)
        print("✅ Prompt已构建")
        print(f"\n完整Prompt:\n{final_prompt}")

        # 步骤4: LLM生成答案
        print("\n步骤4: LLM生成答案")
        print("-" * 60)

        from langchain.schema import HumanMessage
        response = llm.invoke([HumanMessage(content=final_prompt)])

        print(f"✅ 答案: {response.content}")

        # 步骤5: 返回结果
        print("\n步骤5: 返回结果")
        print("-" * 60)
        print(f"最终答案: {response.content}")
        print(f"引用来源: {len(docs)} 个文档")

    except Exception as e:
        print(f"\n❌ 演示失败: {e}")


# ============ 不同检索策略对比 ============

def demo_retrieval_strategies():
    """演示不同的检索策略"""
    print("\n" + "=" * 60)
    print("3. 检索策略对比")
    print("=" * 60)

    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain.vectorstores import Chroma

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        question = "什么是深度学习？"

        # 策略1: 相似度搜索
        print("\n策略1: 相似度搜索 (Similarity Search)")
        print("-" * 60)

        docs = vectorstore.similarity_search(question, k=3)
        print(f"检索到 {len(docs)} 个文档")

        for i, doc in enumerate(docs, 1):
            print(f"\n{i}. {doc.page_content[:100]}...")

        # 策略2: 带分数的相似度搜索
        print("\n\n策略2: 带分数的相似度搜索")
        print("-" * 60)

        docs_with_scores = vectorstore.similarity_search_with_score(question, k=3)
        print(f"检索到 {len(docs_with_scores)} 个文档")

        for i, (doc, score) in enumerate(docs_with_scores, 1):
            print(f"\n{i}. 相似度分数: {score:.4f}")
            print(f"   内容: {doc.page_content[:100]}...")

        # 策略3: MMR (Maximum Marginal Relevance)
        print("\n\n策略3: MMR检索 (多样性)")
        print("-" * 60)

        docs_mmr = vectorstore.max_marginal_relevance_search(question, k=3)
        print(f"检索到 {len(docs_mmr)} 个文档")

        for i, doc in enumerate(docs_mmr, 1):
            print(f"\n{i}. {doc.page_content[:100]}...")

        print("\n💡 策略说明:")
        print("  • Similarity: 返回最相似的文档")
        print("  • With Score: 同时返回相似度分数")
        print("  • MMR: 平衡相关性和多样性")

    except Exception as e:
        print(f"\n❌ 演示失败: {e}")


# ============ 自定义Prompt模板 ============

def demo_custom_prompts():
    """演示自定义Prompt模板"""
    print("\n" + "=" * 60)
    print("4. 自定义Prompt模板")
    print("=" * 60)

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain.vectorstores import Chroma
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 自定义Prompt模板
        custom_template = """你是一个AI助手，专门回答关于人工智能的问题。

请基于以下上下文回答问题。要求：
1. 如果上下文中有答案，请详细回答
2. 如果上下文中没有答案，请明确说明
3. 回答要简洁明了
4. 可以适当补充相关知识

上下文:
{context}

问题: {question}

答案:"""

        PROMPT = PromptTemplate(
            template=custom_template,
            input_variables=["context", "question"]
        )

        # 创建QA链
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        # 测试
        questions = [
            "机器学习有哪些主要方法？",
            "量子计算是什么？",  # 文档中没有的内容
        ]

        for i, question in enumerate(questions, 1):
            print(f"\n问题 {i}: {question}")
            print("-" * 60)

            result = qa_chain.invoke({"query": question})
            print(f"\n答案: {result['result']}")

    except Exception as e:
        print(f"\n❌ 演示失败: {e}")


# ============ RAG系统评估 ============

def demo_rag_evaluation():
    """演示RAG系统评估"""
    print("\n" + "=" * 60)
    print("5. RAG系统评估")
    print("=" * 60)

    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain.vectorstores import Chroma

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        # 测试问题集
        test_cases = [
            {
                "question": "什么是机器学习？",
                "expected_keywords": ["学习", "数据", "算法"]
            },
            {
                "question": "深度学习的应用有哪些？",
                "expected_keywords": ["图像", "语音", "自然语言"]
            },
            {
                "question": "NLP是什么？",
                "expected_keywords": ["自然语言", "处理", "计算机"]
            }
        ]

        print("\n评估指标:")
        print("-" * 60)

        total_score = 0
        for i, test in enumerate(test_cases, 1):
            question = test["question"]
            expected = test["expected_keywords"]

            # 检索文档
            docs = vectorstore.similarity_search(question, k=3)

            # 检查关键词覆盖
            retrieved_text = " ".join([doc.page_content for doc in docs])
            found_keywords = [kw for kw in expected if kw in retrieved_text]

            score = len(found_keywords) / len(expected)
            total_score += score

            print(f"\n测试 {i}: {question}")
            print(f"  期望关键词: {expected}")
            print(f"  找到关键词: {found_keywords}")
            print(f"  覆盖率: {score:.2%}")

        avg_score = total_score / len(test_cases)
        print(f"\n平均覆盖率: {avg_score:.2%}")

        # 检索延迟测试
        print("\n\n检索性能:")
        print("-" * 60)

        import time
        start_time = time.time()

        for _ in range(10):
            vectorstore.similarity_search("测试问题", k=3)

        avg_latency = (time.time() - start_time) / 10

        print(f"平均检索延迟: {avg_latency*1000:.2f}ms")

    except Exception as e:
        print(f"\n❌ 评估失败: {e}")


# ============ 最佳实践 ============

def show_rag_best_practices():
    """显示RAG最佳实践"""
    print("\n" + "=" * 60)
    print("6. RAG最佳实践")
    print("=" * 60)

    best_practices = """
╔══════════════════════════════════════════════════════════════╗
║                    RAG系统构建指南                           ║
╚══════════════════════════════════════════════════════════════╝

1. 文档准备
   ✅ 清理文档格式
   ✅ 去除无关内容
   ✅ 统一编码格式
   ✅ 添加元数据

2. 分块策略
   ✅ 选择合适的chunk_size (推荐500)
   ✅ 设置适当的overlap (推荐10-20%)
   ✅ 保持语义完整性
   ✅ 测试不同配置

3. 向量存储
   ✅ 选择合适的数据库
   ✅ 定期备份
   ✅ 监控存储大小
   ✅ 优化索引性能

4. 检索优化
   ✅ 调整top_k参数 (推荐3-5)
   ✅ 使用相似度阈值过滤
   ✅ 考虑使用MMR增加多样性
   ✅ 添加metadata过滤

5. Prompt工程
   ✅ 清晰的指令
   ✅ 提供上下文
   ✅ 要求引用来源
   ✅ 处理无答案情况

6. 质量保证
   ✅ 建立测试集
   ✅ 评估检索质量
   ✅ 监控答案质量
   ✅ 收集用户反馈

╔══════════════════════════════════════════════════════════════╗
║                    常见问题解决                              ║
╚══════════════════════════════════════════════════════════════╝

Q1: 检索不到相关文档？
A:  • 检查文档是否正确加载
    • 调整chunk_size
    • 增加top_k
    • 检查embedding模型

Q2: 答案不准确？
A:  • 优化Prompt模板
    • 增加检索文档数量
    • 改进文档质量
    • 使用更强的LLM

Q3: 响应太慢？
A:  • 优化向量数据库
    • 减少检索文档数
    • 使用缓存
    • 考虑异步处理

Q4: 成本太高？
A:  • 使用开源embedding
    • 减少LLM调用
    • 优化chunk数量
    • 使用便宜的模型

╔══════════════════════════════════════════════════════════════╗
║                    性能优化建议                              ║
╚══════════════════════════════════════════════════════════════╝

1. 检索优化
   • 使用FAISS而不是Chroma (大规模)
   • 启用GPU加速
   • 使用近似搜索
   • 批量查询

2. 缓存策略
   • 缓存常见问题
   • 缓存embedding结果
   • 使用Redis缓存

3. 并发处理
   • 异步检索
   • 并行处理多个查询
   • 使用连接池

4. 成本控制
   • 监控API调用
   • 设置调用限制
   • 使用本地模型
   • 优化token使用

╔══════════════════════════════════════════════════════════════╗
║                    推荐配置                                  ║
╚══════════════════════════════════════════════════════════════╝

基础配置:
  chunk_size: 500
  chunk_overlap: 50
  top_k: 3
  temperature: 0
  model: gpt-4o-mini

高质量配置:
  chunk_size: 800
  chunk_overlap: 100
  top_k: 5
  temperature: 0
  model: gpt-4o
  + 重排序

高性能配置:
  chunk_size: 300
  chunk_overlap: 30
  top_k: 3
  temperature: 0
  model: gpt-4o-mini
  + FAISS
  + 缓存
"""

    print(best_practices)


def main():
    """主函数"""
    print("\n🔍 基础RAG系统演示\n")

    try:
        # 1. 构建RAG系统
        qa_chain, vectorstore = build_basic_rag()

        if qa_chain is None:
            print("\n⚠️  RAG系统构建失败，跳过后续演示")
            return

        # 2. RAG工作流程
        demo_rag_workflow()

        # 3. 检索策略
        demo_retrieval_strategies()

        # 4. 自定义Prompt
        demo_custom_prompts()

        # 5. 系统评估
        demo_rag_evaluation()

        # 6. 最佳实践
        show_rag_best_practices()

        print("\n" + "=" * 60)
        print("✅ 所有演示完成！")
        print("=" * 60)

        print("\n💡 关键要点:")
        print("1. RAG = 检索 + 生成")
        print("2. 文档质量直接影响答案质量")
        print("3. 合适的chunk_size很重要")
        print("4. 需要优化Prompt模板")
        print("5. 持续评估和优化系统")

        print("\n📊 系统指标:")
        print("• 检索延迟: <100ms")
        print("• 生成延迟: <2s")
        print("• 准确率: >80%")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请检查:")
        print("1. API密钥配置")
        print("2. 依赖包安装")
        print("3. 文档文件存在")


if __name__ == "__main__":
    main()
