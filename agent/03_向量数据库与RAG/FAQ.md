# 常见问题解答 (FAQ)

## 📋 目录
- [Embedding相关](#embedding相关)
- [向量数据库](#向量数据库)
- [文档处理](#文档处理)
- [RAG系统](#rag系统)
- [性能优化](#性能优化)
- [成本控制](#成本控制)

---

## Embedding相关

### Q1: 什么是Embedding？
**A:** Embedding是将文本转换为数值向量的过程。相似的文本在向量空间中距离更近，这使得计算机能够理解文本的语义相似性。

示例：
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query("人工智能")
# 返回1536维的向量
```

---

### Q2: 如何选择Embedding模型？
**A:** 根据需求选择：

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 英文任务 | OpenAI ada-002 | 质量高、多语言 |
| 中文任务 | text2vec-base-chinese | 中文优化 |
| 多语言 | multilingual-e5 | 支持100+语言 |
| 成本敏感 | sentence-transformers | 免费开源 |
| 本地部署 | all-MiniLM-L6-v2 | 轻量快速 |

---

### Q3: Embedding的维度是什么意思？
**A:** 维度是向量的长度。

- OpenAI ada-002: 1536维
- sentence-transformers: 384-768维
- text2vec: 768维

**维度越高**：
- ✅ 表达能力越强
- ❌ 计算和存储成本越高

**选择建议**：
- 一般任务：384-768维足够
- 高精度需求：1024-1536维

---

### Q4: 如何计算文本相似度？
**A:** 最常用的是余弦相似度：

```python
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 相似度范围: -1 到 1
# 越接近1越相似
```

其他方法：
- 欧氏距离：越小越相似
- 点积：越大越相似

---

## 向量数据库

### Q5: Chroma vs FAISS，如何选择？
**A:** 根据场景选择：

**选择Chroma**：
- ✅ 开发和测试阶段
- ✅ 小规模应用（<100K文档）
- ✅ 需要metadata过滤
- ✅ 希望API简单

**选择FAISS**：
- ✅ 生产环境
- ✅ 大规模数据（>100K文档）
- ✅ 性能要求高
- ✅ 需要GPU加速

**选择Pinecone**：
- ✅ 商业项目
- ✅ 需要高可用性
- ✅ 不想管理基础设施
- ❌ 需要付费

---

### Q6: 如何持久化向量数据库？
**A:**

**Chroma（自动持久化）**：
```python
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
# 自动保存
```

**FAISS（手动持久化）**：
```python
import faiss

# 保存索引
faiss.write_index(index, "my_index.faiss")

# 加载索引
index = faiss.read_index("my_index.faiss")

# 保存文档映射
import pickle
with open("documents.pkl", "wb") as f:
    pickle.dump(documents, f)
```

---

### Q7: 向量数据库如何更新？
**A:**

**增量添加**：
```python
# Chroma
vectorstore.add_documents(new_documents)

# FAISS
new_embeddings = get_embeddings(new_documents)
index.add(new_embeddings)
```

**删除文档**：
```python
# Chroma
vectorstore.delete(ids=["doc_1", "doc_2"])

# FAISS
# 需要重建索引（不支持直接删除）
```

**更新文档**：
```python
# Chroma
vectorstore.update_document(
    document_id="doc_1",
    document=new_document
)

# FAISS
# 删除旧的，添加新的
```

---

### Q8: 如何处理大规模数据？
**A:** 优化策略：

1. **使用FAISS**：
   ```python
   # 使用IVF索引（聚类索引）
   quantizer = faiss.IndexFlatL2(dimension)
   index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)
   index.train(training_vectors)
   index.add(vectors)
   ```

2. **批量处理**：
   ```python
   batch_size = 100
   for i in range(0, len(documents), batch_size):
       batch = documents[i:i+batch_size]
       vectorstore.add_documents(batch)
   ```

3. **使用GPU加速**：
   ```bash
   pip install faiss-gpu
   ```

4. **分片存储**：
   - 按类别分片
   - 按时间分片
   - 并行查询

---

## 文档处理

### Q9: 如何选择chunk_size？
**A:** 根据文档类型选择：

| 文档类型 | 推荐大小 | 原因 |
|---------|---------|------|
| 问答对 | 200-300 | 短小精悍 |
| 文章段落 | 500-800 | 保持语义完整 |
| 技术文档 | 800-1200 | 包含完整概念 |
| 书籍章节 | 1000-1500 | 保持上下文 |
| 代码文件 | 300-500 | 保持函数完整 |

**调整建议**：
- 检索不准确 → 调整chunk_size
- 上下文不完整 → 增加chunk_size或overlap
- 成本太高 → 减小chunk_size

---

### Q10: overlap应该设置多少？
**A:**

**推荐值**：10-20% 的chunk_size

```python
chunk_size = 500
chunk_overlap = 50  # 10%

# 或者
chunk_overlap = 100  # 20%
```

**作用**：
- 避免信息在边界处丢失
- 保持上下文连贯性
- 提高检索召回率

**注意**：
- overlap太小：可能丢失信息
- overlap太大：增加存储和成本

---

### Q11: 如何处理PDF文档？
**A:** 使用LangChain的PDF加载器：

```python
from langchain.document_loaders import PyPDFLoader

# 加载PDF
loader = PyPDFLoader("document.pdf")
pages = loader.load()

# 分块
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(pages)
```

**注意事项**：
- PDF可能包含图片和表格
- 需要处理格式问题
- 考虑使用OCR处理扫描PDF

---

### Q12: 如何处理代码文档？
**A:** 使用代码感知的分割策略：

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 针对代码的分割器
code_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    separators=[
        "\n\nclass ",
        "\n\ndef ",
        "\n\n",
        "\n",
        " ",
        ""
    ]
)

chunks = code_splitter.split_text(code_text)
```

**建议**：
- 保持函数/类完整
- 保留缩进结构
- 添加代码语言metadata

---

## RAG系统

### Q13: RAG和Fine-tuning有什么区别？
**A:**

| 维度 | RAG | Fine-tuning |
|------|-----|-------------|
| 知识更新 | 实时更新 | 需要重新训练 |
| 成本 | 低（API调用） | 高（训练成本） |
| 实施难度 | 简单 | 复杂 |
| 可解释性 | 高（可追溯来源） | 低 |
| 准确性 | 依赖检索质量 | 高 |
| 适用场景 | 知识密集型 | 任务特定型 |

**选择建议**：
- 需要频繁更新知识 → RAG
- 需要改变模型行为 → Fine-tuning
- 可以结合使用

---

### Q14: 如何提高RAG的准确率？
**A:** 多方面优化：

**1. 优化文档质量**
- 清理无关内容
- 统一格式
- 添加元数据

**2. 优化分块策略**
- 调整chunk_size
- 增加overlap
- 保持语义完整

**3. 优化检索**
- 使用重排序
- 添加metadata过滤
- 调整top_k
- 使用混合搜索

**4. 优化Prompt**
- 清晰的指令
- 要求引用来源
- 处理无答案情况
- 提供Few-shot示例

**5. 使用高级技巧**
- Query改写
- HyDE
- Multi-Query

---

### Q15: RAG系统响应太慢怎么办？
**A:** 性能优化策略：

**1. 优化检索**
```python
# 使用FAISS代替Chroma
import faiss
index = faiss.IndexFlatL2(dimension)

# 减少检索文档数
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # 从5减到3
)
```

**2. 使用缓存**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query):
    return vectorstore.similarity_search(query)
```

**3. 异步处理**
```python
import asyncio

async def async_rag(question):
    docs = await vectorstore.asimilarity_search(question)
    answer = await llm.ainvoke(prompt)
    return answer
```

**4. 流式输出**
```python
for chunk in qa_chain.stream({"query": question}):
    print(chunk, end="", flush=True)
```

---

### Q16: 如何处理"我不知道"的情况？
**A:** 在Prompt中明确要求：

```python
template = """基于以下上下文回答问题。

重要规则:
1. 只使用上下文中的信息
2. 如果上下文中没有答案，明确说"根据提供的信息，我无法回答这个问题"
3. 不要编造信息

上下文:
{context}

问题: {question}

答案:"""
```

**额外措施**：
- 设置相似度阈值
- 检查检索文档的相关性
- 提供相关问题建议

---

## 性能优化

### Q17: 如何提高检索速度？
**A:**

**1. 使用更快的向量数据库**
```python
# FAISS比Chroma快5-10倍
import faiss
index = faiss.IndexFlatL2(dimension)
```

**2. 优化索引类型**
```python
# 对于大数据量，使用IVF索引
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)
```

**3. 使用GPU加速**
```bash
pip install faiss-gpu
```

**4. 减少检索数量**
```python
# 从k=5减到k=3
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

**5. 批量查询**
```python
# 批量处理多个查询
queries = ["问题1", "问题2", "问题3"]
results = vectorstore.similarity_search_batch(queries)
```

---

### Q18: 内存占用太大怎么办？
**A:** 内存优化策略：

**1. 使用量化**
```python
# FAISS支持向量量化
import faiss
index = faiss.IndexIVFPQ(
    quantizer, dimension, nlist, m, nbits
)
```

**2. 流式处理**
```python
# 不要一次加载所有文档
def process_in_batches(documents, batch_size=100):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        yield batch
```

**3. 定期清理**
```python
# 删除旧的或无用的文档
vectorstore.delete(ids=old_doc_ids)
```

**4. 使用磁盘存储**
```python
# Chroma自动持久化到磁盘
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

---

### Q19: 如何评估RAG系统质量？
**A:** 建立评估体系：

**1. 准备测试集**
```python
test_cases = [
    {
        "question": "什么是机器学习？",
        "expected_answer": "机器学习是...",
        "relevant_docs": ["doc_1", "doc_2"]
    },
    # 更多测试用例...
]
```

**2. 评估检索质量**
```python
# Precision: 检索结果中相关文档的比例
precision = relevant_retrieved / total_retrieved

# Recall: 相关文档中被检索到的比例
recall = relevant_retrieved / total_relevant

# F1 Score
f1 = 2 * precision * recall / (precision + recall)
```

**3. 评估答案质量**
```python
# 使用LLM评估
eval_prompt = f"""评估答案质量（0-10分）：

问题: {question}
答案: {answer}
参考答案: {expected_answer}

评分:"""

score = llm.invoke(eval_prompt)
```

**4. 监控性能指标**
- 检索延迟
- 生成延迟
- 准确率
- 用户满意度

---

## 成本控制

### Q20: RAG系统成本太高怎么办？
**A:** 成本优化策略：

**1. 使用开源Embedding**
```python
# 使用sentence-transformers代替OpenAI
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# 完全免费
```

**2. 使用便宜的LLM**
```python
# 使用gpt-4o-mini代替gpt-4o
llm = ChatOpenAI(model="gpt-4o-mini")
# 成本降低90%
```

**3. 实施缓存**
```python
from langchain.cache import InMemoryCache
import langchain

langchain.llm_cache = InMemoryCache()
# 相同查询不会重复调用API
```

**4. 上下文压缩**
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
# 减少30-50% token
```

**5. 批量处理**
```python
# 批量生成embedding
texts = ["文本1", "文本2", "文本3"]
embeddings = embedding_model.embed_documents(texts)
# 比单个处理更便宜
```

---

### Q21: 如何估算RAG系统成本？
**A:** 成本计算公式：

**Embedding成本**：
```
OpenAI ada-002: $0.0001 / 1K tokens
成本 = (文档tokens / 1000) * $0.0001
```

**LLM成本**：
```
gpt-4o-mini:
  输入: $0.00015 / 1K tokens
  输出: $0.0006 / 1K tokens

每次查询成本 =
  (上下文tokens + 问题tokens) * $0.00015 +
  (答案tokens) * $0.0006
```

**示例计算**：
```
假设:
- 文档: 10,000个，平均500 tokens
- 每次查询: 检索3个文档，生成200 tokens答案

Embedding成本:
  10,000 * 500 / 1000 * $0.0001 = $0.50 (一次性)

每次查询成本:
  (3 * 500 + 50) * $0.00015 + 200 * $0.0006
  = $0.00024 + $0.00012
  = $0.00036

1000次查询: $0.36
```

---

### Q22: 如何监控RAG系统成本？
**A:** 实施监控：

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = qa_chain.invoke({"query": question})

    print(f"Tokens使用: {cb.total_tokens}")
    print(f"成本: ${cb.total_cost:.4f}")
    print(f"调用次数: {cb.successful_requests}")
```

**建议**：
- 记录每次查询的成本
- 设置成本告警
- 定期分析成本报告
- 优化高成本查询

---

## 高级技巧

### Q23: 什么时候应该使用重排序？
**A:**

**适用场景**：
- ✅ 需要高精度
- ✅ 检索结果质量不稳定
- ✅ 成本不敏感
- ✅ 关键业务场景

**不适用场景**：
- ❌ 成本敏感
- ❌ 延迟要求严格
- ❌ 检索质量已经很好

**实施方法**：
```python
# 第一阶段：检索更多候选
docs = vectorstore.similarity_search(query, k=10)

# 第二阶段：重排序
reranked_docs = rerank(query, docs, top_k=3)
```

**效果**：
- 准确率提升：+15-30%
- 成本增加：+50-100%
- 延迟增加：+0.5-1s

---

### Q24: HyDE什么时候有用？
**A:**

**适用场景**：
- 用户查询是问题形式
- 文档是答案形式
- 查询与文档风格差异大

**示例**：
```
查询: "如何使用Python读取文件？"
文档: "使用open()函数可以读取文件..."

问题: 查询和文档的表达方式不同
解决: 用HyDE生成假设答案，用答案检索
```

**实施**：
```python
# 生成假设文档
hyde_doc = llm.invoke(f"回答问题: {query}")

# 用假设文档检索
docs = vectorstore.similarity_search(hyde_doc)
```

**效果**：
- 提升：+15-25% 准确率
- 成本：+1次LLM调用

---

### Q25: 如何实现实时更新？
**A:** 增量更新策略：

**1. 监听文档变化**
```python
import watchdog

def on_document_added(filepath):
    # 加载新文档
    loader = TextLoader(filepath)
    docs = loader.load()

    # 分块
    chunks = splitter.split_documents(docs)

    # 添加到向量存储
    vectorstore.add_documents(chunks)
```

**2. 定期更新**
```python
import schedule

def update_index():
    # 检查新文档
    new_docs = get_new_documents()

    # 更新索引
    vectorstore.add_documents(new_docs)

# 每小时更新一次
schedule.every().hour.do(update_index)
```

**3. 版本管理**
```python
# 为文档添加版本号
metadata = {
    "version": "1.0",
    "updated_at": "2026-01-25"
}

# 更新时删除旧版本
vectorstore.delete(ids=old_version_ids)
vectorstore.add_documents(new_version_docs)
```

---

## 📞 获取更多帮助

### 官方文档
- [LangChain RAG教程](https://python.langchain.com/docs/use_cases/question_answering/)
- [Chroma文档](https://docs.trychroma.com/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)

### 社区资源
- LangChain Discord
- GitHub Discussions
- Stack Overflow

### 提问技巧
1. 提供完整的错误信息
2. 说明你的配置和环境
3. 提供最小可复现示例
4. 说明已尝试的解决方法

---

**最后更新**: 2026-01-25
**版本**: 1.0
