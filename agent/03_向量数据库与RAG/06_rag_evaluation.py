"""
RAG系统评估和优化
学习如何评估和改进RAG系统
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import time
import json

# 加载环境变量
load_dotenv()


# ============ 评估指标 ============

def demo_evaluation_metrics():
    """演示评估指标"""
    print("=" * 60)
    print("1. RAG评估指标")
    print("=" * 60)

    metrics_info = """
╔══════════════════════════════════════════════════════════════╗
║                    RAG评估指标体系                           ║
╚══════════════════════════════════════════════════════════════╝

一、检索质量指标

1. Precision (精确率)
   定义: 检索结果中相关文档的比例
   公式: Precision = 相关文档数 / 检索文档总数
   目标: >0.8

2. Recall (召回率)
   定义: 相关文档中被检索到的比例
   公式: Recall = 检索到的相关文档数 / 总相关文档数
   目标: >0.7

3. F1 Score
   定义: Precision和Recall的调和平均
   公式: F1 = 2 * (Precision * Recall) / (Precision + Recall)
   目标: >0.75

4. MRR (Mean Reciprocal Rank)
   定义: 第一个相关文档位置的倒数的平均值
   公式: MRR = 1/N * Σ(1/rank_i)
   目标: >0.7

5. NDCG (Normalized Discounted Cumulative Gain)
   定义: 考虑位置的排序质量
   目标: >0.8

二、生成质量指标

1. Faithfulness (忠实度)
   定义: 答案是否基于检索的文档
   评估: 检查答案中的信息是否来自文档
   目标: >0.9

2. Answer Relevance (答案相关性)
   定义: 答案是否回答了问题
   评估: 答案与问题的相关程度
   目标: >0.8

3. Context Relevance (上下文相关性)
   定义: 检索的文档是否与问题相关
   评估: 文档与问题的相关程度
   目标: >0.8

三、系统性能指标

1. Latency (延迟)
   • 检索延迟: <100ms
   • 生成延迟: <2s
   • 总延迟: <3s

2. Throughput (吞吐量)
   • 目标: >10 QPS
   • 峰值: >50 QPS

3. Cost (成本)
   • Token成本: <$0.01/query
   • 计算成本: 可控

四、用户体验指标

1. User Satisfaction (用户满意度)
   • 评分: >4/5
   • 采纳率: >70%

2. Task Success Rate (任务成功率)
   • 目标: >80%

3. Response Quality (响应质量)
   • 完整性: >85%
   • 准确性: >90%
"""

    print(metrics_info)


# ============ 自动化评估 ============

def demo_automated_evaluation():
    """演示自动化评估"""
    print("\n" + "=" * 60)
    print("2. 自动化评估")
    print("=" * 60)

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain.vectorstores import Chroma

        # 检查向量数据库
        if not os.path.exists("chroma_db"):
            print("⚠️  请先运行 04_basic_rag.py 创建向量数据库")
            return

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 测试集
        test_cases = [
            {
                "question": "什么是机器学习？",
                "expected_keywords": ["学习", "数据", "算法", "模型"],
                "ground_truth": "机器学习是人工智能的一个分支，使计算机能够从数据中学习"
            },
            {
                "question": "深度学习有哪些应用？",
                "expected_keywords": ["图像", "语音", "自然语言", "识别"],
                "ground_truth": "深度学习在图像识别、语音识别和自然语言处理等领域有应用"
            },
            {
                "question": "NLP是什么？",
                "expected_keywords": ["自然语言", "处理", "计算机", "理解"],
                "ground_truth": "NLP是自然语言处理，使计算机能够理解人类语言"
            }
        ]

        print("\n评估测试集...")
        print("-" * 60)

        results = []

        for i, test in enumerate(test_cases, 1):
            question = test["question"]
            expected_keywords = test["expected_keywords"]

            print(f"\n测试 {i}: {question}")

            # 检索文档
            docs = vectorstore.similarity_search(question, k=3)
            retrieved_text = " ".join([doc.page_content for doc in docs])

            # 计算Precision
            found_keywords = [kw for kw in expected_keywords if kw in retrieved_text]
            precision = len(found_keywords) / len(expected_keywords)

            # 生成答案
            from langchain.schema import HumanMessage
            prompt = f"""基于以下上下文回答问题：

上下文: {retrieved_text}

问题: {question}

答案:"""

            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content

            # 评估答案相关性
            eval_prompt = f"""评估答案是否回答了问题。给出0-1的分数。

问题: {question}
答案: {answer}

只返回数字:"""

            eval_response = llm.invoke([HumanMessage(content=eval_prompt)])
            try:
                relevance = float(eval_response.content.strip())
            except:
                relevance = 0.5

            results.append({
                "question": question,
                "precision": precision,
                "relevance": relevance,
                "answer": answer
            })

            print(f"  关键词覆盖率: {precision:.2%}")
            print(f"  答案相关性: {relevance:.2%}")

        # 总体统计
        print("\n\n总体评估结果:")
        print("=" * 60)

        avg_precision = sum(r["precision"] for r in results) / len(results)
        avg_relevance = sum(r["relevance"] for r in results) / len(results)

        print(f"平均关键词覆盖率: {avg_precision:.2%}")
        print(f"平均答案相关性: {avg_relevance:.2%}")
        print(f"F1 Score: {2 * avg_precision * avg_relevance / (avg_precision + avg_relevance):.2%}")

        return results

    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        return None


# ============ 性能测试 ============

def demo_performance_testing():
    """演示性能测试"""
    print("\n" + "=" * 60)
    print("3. 性能测试")
    print("=" * 60)

    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain.vectorstores import Chroma
        import time

        if not os.path.exists("chroma_db"):
            print("⚠️  请先运行 04_basic_rag.py 创建向量数据库")
            return

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        # 测试查询
        test_queries = [
            "什么是机器学习？",
            "深度学习的应用",
            "NLP技术",
            "计算机视觉",
            "人工智能发展"
        ]

        print("\n1. 检索延迟测试")
        print("-" * 60)

        latencies = []
        for query in test_queries:
            start_time = time.time()
            docs = vectorstore.similarity_search(query, k=3)
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)

        print(f"平均延迟: {avg_latency:.2f}ms")
        print(f"最大延迟: {max_latency:.2f}ms")
        print(f"最小延迟: {min_latency:.2f}ms")

        # 吞吐量测试
        print("\n\n2. 吞吐量测试")
        print("-" * 60)

        num_queries = 20
        start_time = time.time()

        for i in range(num_queries):
            query = test_queries[i % len(test_queries)]
            vectorstore.similarity_search(query, k=3)

        total_time = time.time() - start_time
        qps = num_queries / total_time

        print(f"总查询数: {num_queries}")
        print(f"总时间: {total_time:.2f}秒")
        print(f"吞吐量: {qps:.2f} QPS")

        # 并发测试
        print("\n\n3. 并发性能")
        print("-" * 60)

        import concurrent.futures

        def search_query(query):
            start = time.time()
            vectorstore.similarity_search(query, k=3)
            return time.time() - start

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            start_time = time.time()
            futures = [executor.submit(search_query, q) for q in test_queries * 2]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            total_time = time.time() - start_time

        concurrent_qps = len(results) / total_time
        avg_concurrent_latency = sum(results) / len(results) * 1000

        print(f"并发查询数: {len(results)}")
        print(f"总时间: {total_time:.2f}秒")
        print(f"并发吞吐量: {concurrent_qps:.2f} QPS")
        print(f"平均延迟: {avg_concurrent_latency:.2f}ms")

        # 性能评估
        print("\n\n性能评估:")
        print("-" * 60)

        if avg_latency < 100:
            print("✅ 检索延迟: 优秀 (<100ms)")
        elif avg_latency < 200:
            print("⚠️  检索延迟: 良好 (100-200ms)")
        else:
            print("❌ 检索延迟: 需要优化 (>200ms)")

        if qps > 10:
            print("✅ 吞吐量: 优秀 (>10 QPS)")
        elif qps > 5:
            print("⚠️  吞吐量: 良好 (5-10 QPS)")
        else:
            print("❌ 吞吐量: 需要优化 (<5 QPS)")

    except Exception as e:
        print(f"\n❌ 性能测试失败: {e}")


# ============ A/B测试 ============

def demo_ab_testing():
    """演示A/B测试"""
    print("\n" + "=" * 60)
    print("4. A/B测试")
    print("=" * 60)

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain.vectorstores import Chroma
        from langchain.chains import RetrievalQA

        if not os.path.exists("chroma_db"):
            print("⚠️  请先运行 04_basic_rag.py 创建向量数据库")
            return

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 配置A: 基础配置
        print("\n配置A: 基础配置 (k=3, no rerank)")
        print("-" * 60)

        retriever_a = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain_a = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever_a,
            return_source_documents=True
        )

        # 配置B: 优化配置
        print("\n配置B: 优化配置 (k=5, with filtering)")
        print("-" * 60)

        retriever_b = vectorstore.as_retriever(search_kwargs={"k": 5})
        qa_chain_b = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever_b,
            return_source_documents=True
        )

        # 测试问题
        test_questions = [
            "什么是深度学习？",
            "机器学习有哪些方法？",
            "NLP的应用有哪些？"
        ]

        print("\n\nA/B测试结果:")
        print("=" * 60)

        for i, question in enumerate(test_questions, 1):
            print(f"\n问题 {i}: {question}")
            print("-" * 60)

            # 测试配置A
            start_time = time.time()
            result_a = qa_chain_a.invoke({"query": question})
            time_a = time.time() - start_time

            # 测试配置B
            start_time = time.time()
            result_b = qa_chain_b.invoke({"query": question})
            time_b = time.time() - start_time

            print(f"\n配置A:")
            print(f"  答案: {result_a['result'][:100]}...")
            print(f"  文档数: {len(result_a['source_documents'])}")
            print(f"  耗时: {time_a:.2f}秒")

            print(f"\n配置B:")
            print(f"  答案: {result_b['result'][:100]}...")
            print(f"  文档数: {len(result_b['source_documents'])}")
            print(f"  耗时: {time_b:.2f}秒")

        print("\n\n💡 A/B测试建议:")
        print("  • 定义明确的评估指标")
        print("  • 使用足够的测试样本")
        print("  • 收集用户反馈")
        print("  • 数据驱动决策")

    except Exception as e:
        print(f"\n❌ A/B测试失败: {e}")


# ============ 优化建议 ============

def demo_optimization_recommendations():
    """演示优化建议"""
    print("\n" + "=" * 60)
    print("5. 优化建议")
    print("=" * 60)

    recommendations = """
╔══════════════════════════════════════════════════════════════╗
║                    RAG系统优化指南                           ║
╚══════════════════════════════════════════════════════════════╝

一、检索质量优化

问题: 检索不到相关文档
解决方案:
  1. 调整chunk_size (尝试300-800)
  2. 增加chunk_overlap (10-20%)
  3. 改进文档预处理
  4. 使用更好的embedding模型
  5. 尝试混合搜索

问题: 检索到的文档不够精确
解决方案:
  1. 使用重排序
  2. 添加metadata过滤
  3. 调整top_k参数
  4. 使用Query改写
  5. 实施相似度阈值

问题: 检索结果缺乏多样性
解决方案:
  1. 使用MMR检索
  2. 增加top_k
  3. 调整多样性参数
  4. 使用Multi-Query

二、生成质量优化

问题: 答案不准确
解决方案:
  1. 优化Prompt模板
  2. 增加上下文文档数
  3. 使用更强的LLM
  4. 添加Few-shot示例
  5. 实施答案验证

问题: 答案不完整
解决方案:
  1. 增加检索文档数
  2. 调整chunk_size
  3. 优化文档分块策略
  4. 检查文档完整性

问题: 答案包含幻觉
解决方案:
  1. 降低temperature
  2. 强调基于上下文回答
  3. 添加引用来源
  4. 实施事实检查
  5. 使用更可靠的LLM

三、性能优化

问题: 检索延迟高
解决方案:
  1. 使用FAISS代替Chroma
  2. 优化索引结构
  3. 启用GPU加速
  4. 减少检索文档数
  5. 使用缓存

问题: 生成延迟高
解决方案:
  1. 使用更快的模型
  2. 减少上下文长度
  3. 使用流式输出
  4. 优化Prompt长度
  5. 并行处理

问题: 吞吐量低
解决方案:
  1. 使用异步处理
  2. 批量处理请求
  3. 增加并发数
  4. 优化资源配置
  5. 使用负载均衡

四、成本优化

问题: Token成本高
解决方案:
  1. 使用上下文压缩
  2. 减少检索文档数
  3. 优化Prompt长度
  4. 使用便宜的模型
  5. 实施缓存策略

问题: Embedding成本高
解决方案:
  1. 使用开源模型
  2. 批量生成embedding
  3. 缓存embedding结果
  4. 增量更新索引

问题: 计算成本高
解决方案:
  1. 本地部署
  2. 优化资源使用
  3. 使用Spot实例
  4. 实施自动扩缩容

五、用户体验优化

问题: 响应慢
解决方案:
  1. 使用流式输出
  2. 显示进度提示
  3. 优化性能
  4. 预加载常见查询

问题: 答案不友好
解决方案:
  1. 优化Prompt模板
  2. 添加格式化
  3. 提供引用来源
  4. 支持追问

问题: 错误处理差
解决方案:
  1. 完善错误提示
  2. 提供降级方案
  3. 记录错误日志
  4. 实施重试机制

╔══════════════════════════════════════════════════════════════╗
║                    优化优先级                                ║
╚══════════════════════════════════════════════════════════════╝

高优先级 (立即优化):
  1. 检索质量问题
  2. 答案准确性问题
  3. 严重性能问题
  4. 用户体验问题

中优先级 (计划优化):
  1. 性能优化
  2. 成本优化
  3. 功能增强

低优先级 (持续优化):
  1. 边缘情况处理
  2. 细节优化
  3. 实验性功能

╔══════════════════════════════════════════════════════════════╗
║                    优化流程                                  ║
╚══════════════════════════════════════════════════════════════╝

1. 识别问题
   • 收集用户反馈
   • 分析系统指标
   • 进行测试评估

2. 分析原因
   • 定位问题根源
   • 评估影响范围
   • 确定优化方向

3. 制定方案
   • 设计优化方案
   • 评估成本收益
   • 制定实施计划

4. 实施优化
   • 开发和测试
   • A/B测试验证
   • 灰度发布

5. 监控效果
   • 跟踪关键指标
   • 收集用户反馈
   • 持续迭代优化

╔══════════════════════════════════════════════════════════════╗
║                    监控指标                                  ║
╚══════════════════════════════════════════════════════════════╝

关键指标:
  • 检索精度: >80%
  • 答案准确率: >85%
  • 平均延迟: <3s
  • 用户满意度: >4/5
  • 成本: <$0.01/query

监控频率:
  • 实时: 延迟、错误率
  • 每日: 准确率、成本
  • 每周: 用户满意度
  • 每月: 全面评估
"""

    print(recommendations)


# ============ 评估报告生成 ============

def generate_evaluation_report(results):
    """生成评估报告"""
    print("\n" + "=" * 60)
    print("6. 生成评估报告")
    print("=" * 60)

    if results is None:
        print("⚠️  没有评估结果")
        return

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_cases": len(results),
        "metrics": {
            "avg_precision": sum(r["precision"] for r in results) / len(results),
            "avg_relevance": sum(r["relevance"] for r in results) / len(results)
        },
        "details": results
    }

    # 保存报告
    os.makedirs("reports", exist_ok=True)
    report_file = f"reports/evaluation_{int(time.time())}.json"

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 评估报告已保存到: {report_file}")

    # 显示摘要
    print("\n评估摘要:")
    print("-" * 60)
    print(f"测试用例数: {report['test_cases']}")
    print(f"平均精确率: {report['metrics']['avg_precision']:.2%}")
    print(f"平均相关性: {report['metrics']['avg_relevance']:.2%}")

    # 建议
    print("\n优化建议:")
    print("-" * 60)

    if report['metrics']['avg_precision'] < 0.7:
        print("⚠️  检索精度较低，建议:")
        print("  • 优化chunk策略")
        print("  • 使用重排序")
        print("  • 改进文档质量")

    if report['metrics']['avg_relevance'] < 0.7:
        print("⚠️  答案相关性较低，建议:")
        print("  • 优化Prompt模板")
        print("  • 增加检索文档数")
        print("  • 使用更强的LLM")

    if report['metrics']['avg_precision'] >= 0.8 and report['metrics']['avg_relevance'] >= 0.8:
        print("✅ 系统表现良好！")
        print("  • 继续监控指标")
        print("  • 收集用户反馈")
        print("  • 持续优化")


def main():
    """主函数"""
    print("\n📊 RAG系统评估和优化\n")

    try:
        # 1. 评估指标
        demo_evaluation_metrics()

        # 2. 自动化评估
        results = demo_automated_evaluation()

        # 3. 性能测试
        demo_performance_testing()

        # 4. A/B测试
        demo_ab_testing()

        # 5. 优化建议
        demo_optimization_recommendations()

        # 6. 生成报告
        generate_evaluation_report(results)

        print("\n" + "=" * 60)
        print("✅ 所有演示完成！")
        print("=" * 60)

        print("\n💡 关键要点:")
        print("1. 建立完整的评估体系")
        print("2. 定期进行性能测试")
        print("3. 使用A/B测试验证优化")
        print("4. 数据驱动持续优化")
        print("5. 监控关键指标")

        print("\n📈 评估流程:")
        print("  定义指标 → 收集数据 → 分析问题 →")
        print("  制定方案 → 实施优化 → 验证效果")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请检查:")
        print("1. 向量数据库存在")
        print("2. API密钥配置")
        print("3. 依赖包安装")


if __name__ == "__main__":
    main()
