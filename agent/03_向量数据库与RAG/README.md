# 03 - 向量数据库与RAG

## 项目目标
学习向量数据库的使用，掌握RAG（检索增强生成）系统的构建，实现基于文档的智能问答。

## 学习内容
- ✅ Embedding和向量相似度
- ✅ 向量数据库选择和使用
- ✅ 文档加载和分块策略
- ✅ RAG系统构建
- ✅ 检索优化技巧

## 文件说明

### 1. `requirements.txt`
项目依赖包列表

### 2. `.env.example`
环境变量模板文件

### 3. `01_embedding_basics.py`
Embedding基础：向量表示和相似度计算

### 4. `02_vector_databases.py`
向量数据库使用：Chroma, FAISS对比

### 5. `03_document_loading.py`
文档加载和分块策略

### 6. `04_basic_rag.py`
基础RAG系统实现

### 7. `05_advanced_rag.py`
高级RAG技巧：重排序、混合搜索

### 8. `06_rag_evaluation.py`
RAG系统评估和优化

### 9. `data/`
示例文档数据目录

### 10. `utils/`
工具函数目录

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置API密钥
```bash
cp .env.example .env
# 编辑.env文件，填入API密钥
```

### 3. 准备数据
```bash
# 将你的文档放入data/目录
# 支持格式：txt, pdf, docx, md
```

### 4. 运行示例
```bash
# Embedding基础
python 01_embedding_basics.py

# 向量数据库
python 02_vector_databases.py

# 文档加载
python 03_document_loading.py

# 基础RAG
python 04_basic_rag.py

# 高级RAG
python 05_advanced_rag.py

# RAG评估
python 06_rag_evaluation.py
```

## 核心概念

### 1. Embedding（向量嵌入）
将文本转换为高维向量表示，相似的文本在向量空间中距离更近。

**常用模型**：
- OpenAI: text-embedding-ada-002
- 开源: sentence-transformers
- 中文: text2vec, m3e

### 2. 向量数据库
专门用于存储和检索向量的数据库。

**主流选择**：
- **Chroma**: 轻量级，适合开发和小规模应用
- **FAISS**: Facebook开源，高性能本地检索
- **Pinecone**: 云服务，易用但需付费
- **Milvus**: 企业级，适合大规模生产
- **Weaviate**: 功能丰富，支持混合搜索

### 3. RAG（检索增强生成）
结合信息检索和文本生成，让LLM能够访问外部知识库。

**工作流程**：
```
用户问题 → Embedding → 向量检索 → 获取相关文档 →
构建Prompt → LLM生成 → 返回答案
```

### 4. 文档分块（Chunking）
将长文档切分成小块，便于检索和处理。

**策略**：
- 固定大小分块
- 按段落分块
- 按语义分块
- 递归分块

### 5. 检索策略
- **相似度搜索**: 基于向量距离
- **混合搜索**: 向量 + 关键词
- **重排序**: 二次排序提高精度
- **多查询**: 生成多个查询提高召回

## 向量数据库对比

| 特性 | Chroma | FAISS | Pinecone | Milvus |
|------|--------|-------|----------|--------|
| 部署 | 本地 | 本地 | 云端 | 本地/云端 |
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 性能 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 扩展性 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 成本 | 免费 | 免费 | 付费 | 免费 |
| 适用场景 | 开发/小型 | 本地/中型 | 云端/大型 | 生产/大型 |

## RAG优化技巧

### 1. 文档处理优化
- **Chunk大小**: 通常200-500 tokens
- **Overlap**: 10-20%重叠避免信息丢失
- **Metadata**: 添加元数据便于过滤

### 2. 检索优化
- **Top-K选择**: 通常3-5个文档
- **相似度阈值**: 过滤低相关文档
- **重排序**: 使用Cross-Encoder提高精度

### 3. Prompt优化
- **上下文压缩**: 只保留关键信息
- **引用来源**: 标注信息来源
- **Few-shot示例**: 提供答案格式示例

### 4. 高级技巧
- **Query改写**: 优化用户问题
- **HyDE**: 生成假设文档
- **Self-query**: 自动提取过滤条件
- **Multi-query**: 生成多个查询

## 最佳实践

### 1. 选择合适的Embedding模型
- 英文任务: OpenAI ada-002
- 中文任务: text2vec-base-chinese
- 多语言: multilingual-e5
- 成本敏感: 开源模型

### 2. 优化Chunk策略
```python
# 推荐配置
chunk_size = 500  # tokens
chunk_overlap = 50  # 10%重叠
```

### 3. 向量数据库选择
- 开发阶段: Chroma
- 本地部署: FAISS
- 云端服务: Pinecone
- 大规模生产: Milvus

### 4. 检索参数调优
```python
# 推荐配置
top_k = 4  # 检索文档数
similarity_threshold = 0.7  # 相似度阈值
```

### 5. 成本控制
- 使用开源Embedding模型
- 缓存常见查询结果
- 批量处理文档
- 定期清理无用数据

## 实践项目

### 项目1: 个人知识库问答系统
**功能**：
- 上传个人文档（PDF、TXT、MD）
- 智能问答
- 引用来源

**技术栈**：
- Chroma向量数据库
- OpenAI Embedding
- LangChain RAG

### 项目2: 技术文档助手
**功能**：
- 索引技术文档
- 代码示例检索
- 多语言支持

**技术栈**：
- FAISS向量数据库
- 开源Embedding模型
- 自定义检索策略

### 项目3: 企业知识管理系统
**功能**：
- 多用户支持
- 权限管理
- 高级检索

**技术栈**：
- Milvus向量数据库
- 混合搜索
- 重排序优化

## 常见问题

### Q1: 如何选择Chunk大小？
**A**:
- 短文本（问答）: 200-300 tokens
- 中等文本（文章）: 500-800 tokens
- 长文本（书籍）: 1000-1500 tokens

### Q2: 向量数据库如何选择？
**A**:
- 学习/开发: Chroma
- 个人项目: FAISS
- 商业项目: Pinecone或Milvus

### Q3: 如何提高检索准确度？
**A**:
1. 优化Chunk策略
2. 使用重排序
3. 添加Metadata过滤
4. 尝试混合搜索

### Q4: RAG成本如何控制？
**A**:
1. 使用开源Embedding模型
2. 缓存查询结果
3. 优化Chunk数量
4. 选择合适的LLM

## 性能指标

### 检索质量指标
- **Precision**: 检索结果的准确率
- **Recall**: 相关文档的召回率
- **MRR**: 平均倒数排名
- **NDCG**: 归一化折损累积增益

### 系统性能指标
- **检索延迟**: <100ms
- **生成延迟**: <2s
- **吞吐量**: >10 QPS
- **准确率**: >80%

## 下一步

完成本章节后，进入 `04_高级Agent开发` 学习Multi-Agent系统。

---
**创建日期**：2026-01-25
**状态**：进行中
**预计完成**：5-7天
