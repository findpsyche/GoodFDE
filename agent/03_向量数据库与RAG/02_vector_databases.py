"""
向量数据库使用演示
对比Chroma和FAISS的使用
"""

import os
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import time

# 加载环境变量
load_dotenv()


# ============ Chroma数据库演示 ============

def demo_chroma():
    """演示Chroma向量数据库"""
    print("=" * 60)
    print("1. Chroma向量数据库演示")
    print("=" * 60)

    try:
        import chromadb
        from chromadb.config import Settings
        from openai import OpenAI

        # 初始化OpenAI客户端
        client = OpenAI()

        # 创建Chroma客户端
        chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))

        # 创建或获取集合
        collection_name = "demo_collection"

        # 重置集合（如果存在）
        try:
            chroma_client.delete_collection(collection_name)
        except:
            pass

        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"description": "演示集合"}
        )

        print(f"\n✅ 创建集合: {collection_name}")

        # 准备文档
        documents = [
            "Python是一种高级编程语言，广泛用于数据科学和AI开发",
            "机器学习是人工智能的一个重要分支",
            "深度学习使用神经网络来学习数据中的模式",
            "自然语言处理让计算机能够理解人类语言",
            "计算机视觉使机器能够理解和分析图像",
        ]

        # 生成embeddings
        print("\n生成文档embeddings...")
        embeddings = []
        for doc in documents:
            response = client.embeddings.create(
                input=doc,
                model="text-embedding-ada-002"
            )
            embeddings.append(response.data[0].embedding)
            time.sleep(0.5)

        # 添加文档到集合
        print("\n添加文档到Chroma...")
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(documents))],
            metadatas=[{"source": f"document_{i}"} for i in range(len(documents))]
        )

        print(f"✅ 已添加 {len(documents)} 个文档")

        # 查询
        query = "什么是深度学习？"
        print(f"\n查询: {query}")
        print("-" * 60)

        # 生成查询embedding
        query_response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = query_response.data[0].embedding

        # 执行查询
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        print("\n检索结果:")
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
            print(f"\n{i}. 距离: {distance:.4f}")
            print(f"   文档: {doc}")

        # 统计信息
        print("\n" + "-" * 60)
        print("集合统计:")
        print(f"  文档数量: {collection.count()}")
        print(f"  集合名称: {collection.name}")

        return collection

    except ImportError:
        print("\n❌ 需要安装chromadb: pip install chromadb")
        return None
    except Exception as e:
        print(f"\n❌ Chroma演示失败: {e}")
        return None


# ============ FAISS数据库演示 ============

def demo_faiss():
    """演示FAISS向量数据库"""
    print("\n" + "=" * 60)
    print("2. FAISS向量数据库演示")
    print("=" * 60)

    try:
        import faiss
        from openai import OpenAI

        # 初始化OpenAI客户端
        client = OpenAI()

        # 准备文档
        documents = [
            "Python是一种高级编程语言，广泛用于数据科学和AI开发",
            "机器学习是人工智能的一个重要分支",
            "深度学习使用神经网络来学习数据中的模式",
            "自然语言处理让计算机能够理解人类语言",
            "计算机视觉使机器能够理解和分析图像",
        ]

        print("\n生成文档embeddings...")
        embeddings = []
        for doc in documents:
            response = client.embeddings.create(
                input=doc,
                model="text-embedding-ada-002"
            )
            embeddings.append(response.data[0].embedding)
            time.sleep(0.5)

        # 转换为numpy数组
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]

        print(f"\n向量维度: {dimension}")
        print(f"文档数量: {len(documents)}")

        # 创建FAISS索引
        print("\n创建FAISS索引...")

        # 使用L2距离的平面索引
        index = faiss.IndexFlatL2(dimension)

        # 添加向量
        index.add(embeddings_array)

        print(f"✅ 已添加 {index.ntotal} 个向量")

        # 查询
        query = "什么是深度学习？"
        print(f"\n查询: {query}")
        print("-" * 60)

        # 生成查询embedding
        query_response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = np.array([query_response.data[0].embedding]).astype('float32')

        # 执行搜索
        k = 3  # 返回top-3结果
        distances, indices = index.search(query_embedding, k)

        print("\n检索结果:")
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
            print(f"\n{i}. 距离: {distance:.4f}")
            print(f"   文档: {documents[idx]}")

        # 保存索引
        print("\n保存FAISS索引...")
        faiss.write_index(index, "faiss_demo.index")
        print("✅ 索引已保存到 faiss_demo.index")

        # 加载索引
        print("\n加载FAISS索引...")
        loaded_index = faiss.read_index("faiss_demo.index")
        print(f"✅ 索引已加载，包含 {loaded_index.ntotal} 个向量")

        return index, documents

    except ImportError:
        print("\n❌ 需要安装faiss: pip install faiss-cpu")
        return None, None
    except Exception as e:
        print(f"\n❌ FAISS演示失败: {e}")
        return None, None


# ============ 性能对比 ============

def demo_performance_comparison():
    """对比Chroma和FAISS的性能"""
    print("\n" + "=" * 60)
    print("3. 性能对比")
    print("=" * 60)

    try:
        import chromadb
        import faiss
        from openai import OpenAI

        client = OpenAI()

        # 准备更多文档
        print("\n准备测试数据...")
        documents = [
            f"这是测试文档 {i}，包含一些随机内容用于性能测试。"
            for i in range(100)
        ]

        # 生成embeddings
        print("生成embeddings...")
        embeddings = []
        for i, doc in enumerate(documents):
            if i % 20 == 0:
                print(f"  进度: {i}/{len(documents)}")
            response = client.embeddings.create(
                input=doc,
                model="text-embedding-ada-002"
            )
            embeddings.append(response.data[0].embedding)
            time.sleep(0.1)

        # Chroma性能测试
        print("\n测试Chroma性能...")
        chroma_client = chromadb.Client()

        try:
            chroma_client.delete_collection("perf_test")
        except:
            pass

        collection = chroma_client.create_collection("perf_test")

        start_time = time.time()
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
        chroma_insert_time = time.time() - start_time

        # 查询性能
        query_embedding = embeddings[0]
        start_time = time.time()
        for _ in range(10):
            collection.query(query_embeddings=[query_embedding], n_results=5)
        chroma_query_time = (time.time() - start_time) / 10

        # FAISS性能测试
        print("\n测试FAISS性能...")
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]

        index = faiss.IndexFlatL2(dimension)

        start_time = time.time()
        index.add(embeddings_array)
        faiss_insert_time = time.time() - start_time

        # 查询性能
        query_array = np.array([query_embedding]).astype('float32')
        start_time = time.time()
        for _ in range(10):
            index.search(query_array, 5)
        faiss_query_time = (time.time() - start_time) / 10

        # 显示结果
        print("\n" + "=" * 60)
        print("性能对比结果 (100个文档)")
        print("=" * 60)

        print("\n插入性能:")
        print(f"  Chroma: {chroma_insert_time:.4f}秒")
        print(f"  FAISS:  {faiss_insert_time:.4f}秒")
        print(f"  速度比: {chroma_insert_time/faiss_insert_time:.2f}x")

        print("\n查询性能 (平均):")
        print(f"  Chroma: {chroma_query_time*1000:.2f}ms")
        print(f"  FAISS:  {faiss_query_time*1000:.2f}ms")
        print(f"  速度比: {chroma_query_time/faiss_query_time:.2f}x")

    except Exception as e:
        print(f"\n❌ 性能对比失败: {e}")


# ============ 高级功能演示 ============

def demo_advanced_features():
    """演示高级功能"""
    print("\n" + "=" * 60)
    print("4. 高级功能演示")
    print("=" * 60)

    try:
        import chromadb
        from openai import OpenAI

        client = OpenAI()
        chroma_client = chromadb.Client()

        # 创建集合
        try:
            chroma_client.delete_collection("advanced_demo")
        except:
            pass

        collection = chroma_client.create_collection("advanced_demo")

        # 准备带metadata的文档
        documents = [
            "Python编程语言基础教程",
            "机器学习算法详解",
            "深度学习实战项目",
            "数据科学入门指南",
            "人工智能应用案例",
        ]

        metadatas = [
            {"category": "编程", "level": "初级", "language": "Python"},
            {"category": "AI", "level": "中级", "language": "Python"},
            {"category": "AI", "level": "高级", "language": "Python"},
            {"category": "数据", "level": "初级", "language": "Python"},
            {"category": "AI", "level": "中级", "language": "通用"},
        ]

        # 生成embeddings并添加
        print("\n添加带metadata的文档...")
        embeddings = []
        for doc in documents:
            response = client.embeddings.create(
                input=doc,
                model="text-embedding-ada-002"
            )
            embeddings.append(response.data[0].embedding)
            time.sleep(0.5)

        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )

        # 使用metadata过滤
        print("\n使用metadata过滤查询...")
        print("-" * 60)

        query = "人工智能"
        query_response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = query_response.data[0].embedding

        # 查询1: 只查询AI类别
        print("\n查询1: 只查询AI类别的文档")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where={"category": "AI"}
        )

        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            print(f"\n{i}. {doc}")
            print(f"   元数据: {meta}")

        # 查询2: 查询初级难度
        print("\n查询2: 只查询初级难度的文档")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where={"level": "初级"}
        )

        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            print(f"\n{i}. {doc}")
            print(f"   元数据: {meta}")

        # 更新文档
        print("\n\n更新文档...")
        collection.update(
            ids=["doc_0"],
            documents=["Python编程语言高级教程"],
            metadatas=[{"category": "编程", "level": "高级", "language": "Python"}]
        )
        print("✅ 文档已更新")

        # 删除文档
        print("\n删除文档...")
        collection.delete(ids=["doc_4"])
        print("✅ 文档已删除")

        print(f"\n当前文档数量: {collection.count()}")

    except Exception as e:
        print(f"\n❌ 高级功能演示失败: {e}")


# ============ 最佳实践建议 ============

def show_best_practices():
    """显示最佳实践建议"""
    print("\n" + "=" * 60)
    print("5. 最佳实践建议")
    print("=" * 60)

    best_practices = """
╔══════════════════════════════════════════════════════════════╗
║                    向量数据库选择指南                        ║
╚══════════════════════════════════════════════════════════════╝

1. Chroma
   ✅ 优点:
      • 易于使用，API简洁
      • 支持metadata过滤
      • 自动持久化
      • 适合快速原型开发

   ❌ 缺点:
      • 性能不如FAISS
      • 不适合大规模数据

   💡 适用场景:
      • 开发和测试
      • 小型应用（<100K文档）
      • 需要metadata过滤

2. FAISS
   ✅ 优点:
      • 极高的检索性能
      • 支持GPU加速
      • 多种索引类型
      • 适合大规模数据

   ❌ 缺点:
      • API相对复杂
      • 需要手动管理metadata
      • 需要手动持久化

   💡 适用场景:
      • 生产环境
      • 大规模数据（>100K文档）
      • 性能要求高

3. Pinecone
   ✅ 优点:
      • 完全托管，无需维护
      • 自动扩展
      • 高可用性

   ❌ 缺点:
      • 需要付费
      • 数据在云端

   💡 适用场景:
      • 商业项目
      • 需要高可用性
      • 不想管理基础设施

4. Milvus
   ✅ 优点:
      • 企业级功能
      • 高性能
      • 支持多种索引

   ❌ 缺点:
      • 部署复杂
      • 学习曲线陡

   💡 适用场景:
      • 大型企业应用
      • 需要高级功能
      • 有专业运维团队

╔══════════════════════════════════════════════════════════════╗
║                    性能优化建议                              ║
╚══════════════════════════════════════════════════════════════╝

1. 索引选择
   • 小数据量（<10K）: Flat索引
   • 中等数据量（10K-1M）: IVF索引
   • 大数据量（>1M）: HNSW索引

2. 批量操作
   • 批量插入而不是单条插入
   • 批量查询提高吞吐量

3. 向量维度
   • 权衡精度和性能
   • 考虑降维（PCA）

4. 缓存策略
   • 缓存常见查询
   • 使用LRU缓存

5. 硬件优化
   • 使用SSD存储
   • 考虑GPU加速（FAISS）
   • 增加内存

╔══════════════════════════════════════════════════════════════╗
║                    成本优化建议                              ║
╚══════════════════════════════════════════════════════════════╝

1. Embedding成本
   • 使用开源模型（sentence-transformers）
   • 批量生成embedding
   • 缓存embedding结果

2. 存储成本
   • 定期清理无用数据
   • 使用压缩
   • 选择合适的索引类型

3. 计算成本
   • 本地部署（Chroma/FAISS）
   • 避免频繁重建索引
   • 使用增量更新

╔══════════════════════════════════════════════════════════════╗
║                    决策流程图                                ║
╚══════════════════════════════════════════════════════════════╝

                    开始
                     │
                     ▼
              数据量多大？
                 ╱   ╲
            <10K      >10K
              ╱          ╲
             ▼            ▼
        需要metadata？  需要云服务？
          ╱   ╲         ╱   ╲
        是      否     是      否
       ╱          ╲   ╱          ╲
      ▼            ▼ ▼            ▼
   Chroma      FAISS Pinecone   Milvus
"""

    print(best_practices)


def main():
    """主函数"""
    print("\n🗄️  向量数据库使用演示\n")

    try:
        # 1. Chroma演示
        demo_chroma()

        # 2. FAISS演示
        demo_faiss()

        # 3. 性能对比
        demo_performance_comparison()

        # 4. 高级功能
        demo_advanced_features()

        # 5. 最佳实践
        show_best_practices()

        print("\n" + "=" * 60)
        print("✅ 所有演示完成！")
        print("=" * 60)

        print("\n💡 关键要点:")
        print("1. Chroma适合开发和小型应用")
        print("2. FAISS适合生产和大规模应用")
        print("3. 选择合适的索引类型很重要")
        print("4. Metadata过滤可以提高检索精度")
        print("5. 批量操作可以提高性能")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请检查:")
        print("1. 是否安装了chromadb和faiss-cpu")
        print("2. API密钥配置")
        print("3. 网络连接")


if __name__ == "__main__":
    main()
