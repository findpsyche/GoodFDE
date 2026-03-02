# 面试知识清单 · LLM × Web2 系统开发（工程化实战版）

> **定位**：不只是知识清单，而是"让企业系统真正运行起来"的实战指南
> **岗位**：AI 平台研发（兼 Web 全栈）
> **核心理念**：站在生产可用的高度——怎么部署、怎么监控、故障怎么处理、如何验证可用

---

# 一、全局生态位理解（先搞清楚你要交付什么）

## 1.1 系统架构全景图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           用户/业务方                                    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ↓ HTTPS/WebSocket
┌─────────────────────────────────────────────────────────────────────────┐
│                        Nginx 反向代理层                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │
│  │   静态资源   │  │   API 路由   │  │   SSL 终止   │                     │
│  │   (前端)     │  │   (/api/*)   │  │   (443→80)   │                     │
│  └─────────────┘  └──────┬──────┘  └─────────────┘                     │
└───────────────────────────┼──────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          应用服务层                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   FastAPI 后端服务                               │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │    │
│  │  │ 认证中间件│ │ 限流中间件│ │ 日志中间件│ │ 异常处理  │           │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │    │
│  │  ┌──────────────────────────────────────────────────────────┐   │    │
│  │  │                    业务路由层                             │   │    │
│  │  │  /api/chat    /api/rag    /api/agent    /api/admin       │   │    │
│  │  └──────────────────────────────────────────────────────────┘   │    │
│  │  ┌──────────────────────────────────────────────────────────┐   │    │
│  │  │                    服务层                                 │   │    │
│  │  │  LLMService  │  RAGService  │  AgentService  │  AuthService│   │    │
│  │  └──────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                            ↕                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   消息队列层 (Celery + Redis)                    │    │
│  │         异步任务：文档索引、长时 Agent、数据导出                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          数据持久层                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  PostgreSQL  │  │    Redis     │  │  向量数据库   │                  │
│  │  (pgvector)  │  │  (缓存/队列)  │  │ (Milvus/pg)  │                  │
│  │  对话历史    │  │  Session/    │  │  文档Embed    │                  │
│  │  用户数据    │  │  限流/锁     │  │  RAG检索      │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        外部依赖层                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  LLM API │  │ OSS 存储 │  │ 业务API  │  │ 监控告警  │              │
│  │(OpenAI等)│  │(文件/图片)│  │(水利数据)│  │(Prometheus│              │
│  └──────────┘  └──────────┘  └──────────┘  │   +Grafana)│              │
│                                            └──────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1.2 第一性原理：这个系统的本质是什么？

**本质**：把水利领域的知识和任务，通过 LLM 的语言理解能力包一层，变成可交互的智能服务。

**核心挑战**：
1. **准确性**：水利规范涉及安全标准，不能出错 → 需要可追溯的 RAG
2. **可用性**：业务人员不是技术人员 → 需要 Web UI + 流式交互
3. **可靠性**：生产环境不能挂 → 需要监控、容错、降级
4. **可维护**：代码要能长期演进 → 需要工程化规范

## 1.3 你要交付的价值

| 层面 | 交付内容 | 验收标准 |
|------|----------|----------|
| 功能 | API 能调通 LLM/RAG/Agent | 单元测试 + 接口测试通过 |
| 可用 | Web 界面能实际使用 | 端到端测试通过，非研发人员能操作 |
| 稳定 | 服务不挂，响应可接受 | 7×24小时运行，P95延迟 < 3s |
| 可控 | 出问题能知道、能排查 | 日志完整，监控告警配置 |
| 可维护 | 代码清晰，好迭代 | 代码审查通过，文档完整 |

---

# 二、LLM 应用开发核心（生产级）

## 2.1 大模型调用原理与工程化

### 2.1.1 基础调用与错误处理

**第一性原理**：LLM API 调用本质是 HTTP 请求，可能失败、超时、被限流，必须做好容错。

```python
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,  # 支持国产模型兼容接口
            timeout=30.0,  # 超时设置
            max_retries=0,  # 自己控制重试逻辑
        )

    @retry(
        stop=stop_after_attempt(3),  # 最多重试3次
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避
        retry=retry_if_exception_type((openai.APITimeoutError, openai.APIConnectionError)),
    )
    async def chat(
        self,
        messages: list[dict],
        model: str = "gpt-4o",
        temperature: float = 0.2,  # 生产环境用低温度
        max_tokens: int = 2000,
        stream: bool = False,
    ) -> openai.types.ChatCompletion | AsyncStream:
        """
        带重试和监控的 LLM 调用
        """
        import time
        start = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )

            # 记录调用成功
            duration = time.time() - start
            tokens = response.usage.total_tokens if not stream else None
            logger.info(
                "llm_call_success",
                extra={
                    "model": model,
                    "duration_ms": duration * 1000,
                    "tokens": tokens,
                    "temperature": temperature,
                }
            )

            return response

        except openai.RateLimitError as e:
            logger.error(f"LLM rate limit: {e}")
            # 降级策略：切换备用模型或排队
            raise
        except openai.APIError as e:
            logger.error(f"LLM API error: {e}")
            raise
        except Exception as e:
            logger.exception(f"LLM unexpected error: {e}")
            raise
```

**为什么这样设计？**
- **异步调用**：FastAPI 是异步框架，同步调用会阻塞事件循环
- **重试策略**：网络抖动时自动重试，但限流错误不重试（会加重问题）
- **超时控制**：防止请求卡死，占用资源
- **结构化日志**：便于监控和分析

### 2.1.2 流式输出工程化

```python
from fastapi.responses import StreamingResponse
import json
import asyncio

async def chat_stream(request: ChatRequest):
    """流式 SSE 响应，带错误处理和心跳"""

    async def generate():
        try:
            # 发送心跳保持连接
            heartbeat_task = asyncio.create_task(send_heartbeat())

            async for chunk in llm_service.chat_stream(
                messages=request.messages,
                model=request.model,
            ):
                # 解析 chunk
                if chunk.choices and chunk.choices[0].delta.content:
                    data = {
                        "type": "content",
                        "text": chunk.choices[0].delta.content,
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            # 发送结束标记
            yield "data: [DONE]\n\n"

        except Exception as e:
            # 错误也要发给前端
            error_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            heartbeat_task.cancel()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        }
    )

async def send_heartbeat():
    """每15秒发一个注释，防止代理超时"""
    while True:
        await asyncio.sleep(15)
        yield ": heartbeat\n\n"
```

### 2.1.3 多模型切换与降级

```python
class ModelRouter:
    """智能模型路由：成本、速度、质量平衡"""

    def __init__(self):
        # 模型配置：成本、延迟、适用场景
        self.models = {
            "gpt-4o": {"cost": 5, "speed": 3, "quality": 5, "for": "复杂推理"},
            "gpt-4o-mini": {"cost": 1, "speed": 5, "quality": 3, "for": "简单问答"},
            "deepseek-chat": {"cost": 0.5, "speed": 4, "quality": 4, "for": "代码/中文"},
            "qwen-turbo": {"cost": 0.3, "speed": 5, "quality": 3.5, "for": "快速响应"},
        }
        self.primary = "gpt-4o"
        self.fallback = "deepseek-chat"

    async def call_with_fallback(self, messages: list, required_quality: int = 4):
        """带降级的调用"""
        # 根据任务复杂度选择模型
        estimated_complexity = self._estimate_complexity(messages)

        if estimated_complexity < 3:
            model = "gpt-4o-mini"  # 简单任务用便宜模型
        else:
            model = self.primary

        try:
            return await self.llm_service.chat(messages, model=model)
        except Exception as e:
            logger.warning(f"Primary model {model} failed, trying fallback")
            return await self.llm_service.chat(messages, model=self.fallback)
```

## 2.2 Prompt Engineering（生产级）

### 2.2.1 Prompt 管理系统

**不要硬编码 Prompt！** 用配置管理版本。

```python
# prompts.py - 统一管理所有 Prompt
from enum import Enum

class PromptTemplate(Enum):
    """Prompt 模板集中管理，支持版本"""

    WATER_ENGINEER = {
        "version": "v2",
        "system": """你是水利规范查询助手，严格遵循以下原则：

1. 只回答水利专业相关问题
2. 引用规范必须注明：标准号 + 条文号
3. 不确定时明确说"规范中未找到相关内容"，绝不编造
4. 输出 JSON 格式

示例输出：
{{
  "answer": "根据 GB 50286-2013 第 3.1.2 条...",
  "sources": ["GB 50286-2013 第3.1.2条"],
  "confidence": 0.95
}}""",
        "temperature": 0.1,  # 低温度保证稳定
        "max_tokens": 1500,
    }

    DATA_ANALYST = {
        "version": "v1",
        "system": """你是水文数据分析助手...""",
        "temperature": 0.3,
    }

class PromptManager:
    """Prompt 管理器：支持 A/B 测试和灰度"""

    def __init__(self, config_service):
        self.config = config_service

    async def get_prompt(self, name: str, user_id: str = None) -> dict:
        """获取 Prompt，支持用户分组灰度"""
        template = PromptTemplate[name].value

        # A/B 测试：10% 用户用新版本
        if user_id and self._should_use_v2(user_id):
            template = self._get_v2_variant(name)

        return template

    def _should_use_v2(self, user_id: str) -> bool:
        """简单的用户分组算法"""
        return hash(user_id) % 10 == 0
```

### 2.2.2 结构化输出实战

```python
from pydantic import BaseModel, Field
from typing import List, Optional
import instructor

# 方法1：instructor 库（推荐，更灵活）
class WaterQueryResponse(BaseModel):
    """水利规范查询响应结构"""
    answer: str = Field(description="答案，200-500字")
    sources: List[str] = Field(description="引用的标准号和条文号")
    confidence: float = Field(description="置信度 0-1", ge=0, le=1)
    related_concepts: List[str] = Field(description="相关概念，用于推荐", default=[])

# 启用 instructor
client = instructor.from_openai(openai.AsyncOpenAI())

async def query_with_structure(question: str) -> WaterQueryResponse:
    response = await client.chat.completions.create(
        model="gpt-4o",
        response_model=WaterQueryResponse,
        messages=[
            {"role": "system", "content": PromptTemplate.WATER_ENGINEER.value["system"]},
            {"role": "user", "content": question},
        ],
        temperature=0.1,
    )
    return response  # 已经是 WaterQueryResponse 对象

# 方法2：OpenAI 原生 structured outputs（更严格）
async def query_native(question: str) -> WaterQueryResponse:
    response = await client.beta.chat.completions.parse(
        model="gpt-4o",
        response_format=WaterQueryResponse,
        messages=[...],
    )
    return response.parsed
```

### 2.2.3 Prompt 调试方法论

**Prompt 调试 = 代码调试**

```python
class PromptDebugger:
    """Prompt 调试工具"""

    def __init__(self, log_dir: str = "./prompts_debug"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def log_interaction(
        self,
        prompt_name: str,
        messages: list,
        response: str,
        metadata: dict,
    ):
        """记录每次交互，用于分析"""
        import uuid
        import json

        trace_id = str(uuid.uuid4())
        log_file = self.log_dir / f"{prompt_name}_{trace_id}.json"

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump({
                "trace_id": trace_id,
                "prompt_name": prompt_name,
                "messages": messages,
                "response": response,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)

    def analyze_failure(self, trace_id: str):
        """分析失败案例"""
        # 加载失败案例
        # 用 LLM 总结问题
        # 生成改进建议
        pass
```

## 2.3 RAG（检索增强生成）完整工程

### 2.3.1 RAG 系统架构设计

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           离线构建阶段                                   │
│  [原始文档] → [解析] → [分块] → [Embedding] → [向量库]                   │
│              ↓         ↓          ↓                                        │
│           元数据提取   Chunk管理   批量调用                                 │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                           在线检索阶段                                   │
│  [用户问题] → [Query理解] → [检索策略] → [重排] → [上下文构建]            │
│                  ↓           ↓           ↓            ↓                  │
│               意图识别    混合检索     CrossEncoder  截断/填充             │
│                           向量+BM25                                       │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                           生成阶段                                       │
│  [问题+上下文] → [LLM生成] → [验证] → [输出]                              │
│                       ↓           ↓                                       │
│                   Prompt模板    事后检查                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3.2 完整 RAG 实现

```python
from typing import List, Optional
import numpy as np
from sentence_transformers import CrossEncoder

class RAGPipeline:
    """生产级 RAG 管道"""

    def __init__(
        self,
        vector_store,
        reranker: CrossEncoder,
        bm25_index,
        config: dict,
    ):
        self.vector_store = vector_store
        self.reranker = reranker
        self.bm25 = bm25_index
        self.config = config

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> List[dict]:
        """
        混合检索 + 重排
        """
        # 1. 向量检索
        vector_results = await self._vector_search(query, top_k=20, filters=filters)

        # 2. BM25 关键词检索
        bm25_results = self._bm25_search(query, top_k=20)

        # 3. RRF 融合 (Reciprocal Rank Fusion)
        fused = self._rrf_fusion(
            vector_results,
            bm25_results,
            k=60,  # RRF 参数
        )

        # 4. 重排
        reranked = await self._rerank(query, fused[:20], top_k=top_k)

        # 5. 多样性控制（避免重复段落）
        diverse = self._diversify_results(reranked)

        return diverse

    def _rrf_fusion(self, list1: List, list2: List, k: int = 60) -> List:
        """
        RRF 融合算法
        score = sum(1 / (k + rank))
        """
        scores = {}

        for rank, item in enumerate(list1):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

        for rank, item in enumerate(list2):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

        # 按分数排序
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item["doc"] for item, score in sorted_items]

    async def _rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int = 5,
    ) -> List[dict]:
        """CrossEncoder 重排"""
        pairs = [(query, doc["content"]) for doc in candidates]
        scores = self.reranker.predict(pairs)

        # 按分数排序
        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)

        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    def _diversify_results(self, results: List[dict], threshold: float = 0.85) -> List:
        """MMR (Maximal Marginal Relevance) 多样性控制"""
        if len(results) <= 1:
            return results

        selected = [results[0]]
        remaining = results[1:]

        while remaining and len(selected) < self.config["max_results"]:
            # 计算每个候选与已选的最大相似度
            best_idx = 0
            best_score = -float("inf")

            for i, candidate in enumerate(remaining):
                # 相关性 - 相似度
                relevance = candidate["rerank_score"]
                max_similarity = max([
                    self._similarity(candidate, s)
                    for s in selected
                ])
                mmr_score = relevance - 0.5 * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_score > 0:
                selected.append(remaining.pop(best_idx))
            else:
                break

        return selected

    def _similarity(self, doc1: dict, doc2: dict) -> float:
        """计算文档相似度（用 embedding）"""
        # 实际用 cosine similarity
        return np.dot(doc1["embedding"], doc2["embedding"])
```

### 2.3.3 RAG 质量评估

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

class RAGEvaluator:
    """RAG 系统评估"""

    async def evaluate(
        self,
        test_dataset: List[dict],  # {question, ground_truth, context}
        pipeline: RAGPipeline,
    ) -> dict:
        """
        评估 RAG 系统质量
        """
        results = []

        for item in test_dataset:
            # 1. 检索
            retrieved = await pipeline.retrieve(item["question"])

            # 2. 生成
            answer = await llm_service.generate(
                question=item["question"],
                context=retrieved,
            )

            results.append({
                "question": item["question"],
                "answer": answer,
                "contexts": [r["content"] for r in retrieved],
                "ground_truth": item.get("ground_truth"),
            })

        # 3. 用 RAGAS 评估
        from datasets import Dataset
        dataset = Dataset.from_list(results)

        scores = evaluate(
            dataset,
            metrics=[
                faithfulness,      # 幻觉率
                answer_relevancy,  # 答案相关性
                context_precision, # 检索精确度
                context_recall,    # 检索召回率
            ]
        )

        return {
            "faithfulness": scores["faithfulness"],
            "answer_relevancy": scores["answer_relevancy"],
            "context_precision": scores["context_precision"],
            "context_recall": scores["context_recall"],
        }

    def hit_rate(self, retrieved: List, relevant_ids: set) -> float:
        """Hit Rate: 正确答案是否在检索结果中"""
        return int(any(r["id"] in relevant_ids for r in retrieved)) / len(retrieved)

    def mrr(self, retrieved: List, relevant_id: str) -> float:
        """Mean Reciprocal Rank: 正确答案的排名倒数"""
        for i, r in enumerate(retrieved):
            if r["id"] == relevant_id:
                return 1 / (i + 1)
        return 0
```

### 2.3.4 RAG 常见问题与解决方案

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| 召回不相关 | Chunk 策略不当 | 调整 chunk_size，尝试语义分块 |
| 专业术语效果差 | Embedding 模型不匹配 | 用领域模型微调或换 bge-m3 |
| 答案不在文档中 | 幻觉 | Prompt 明确约束 + 事后验证 |
| 跨段落信息丢失 | Chunk 太小 | 增大 overlap 或用滑动窗口 |
| 检索速度慢 | 向量库未优化 | 索引优化、缓存热门查询 |
| 表格/图片丢失 | 解析不完整 | 用专用解析器（TableTransformer） |

## 2.4 Function Calling / Agent 工程化

### 2.4.1 工具调用完整框架

```python
from typing import TypedDict, List, Optional
from enum import Enum
import json

class ToolResult(TypedDict):
    success: bool
    data: Optional[dict]
    error: Optional[str]

class ToolExecutor:
    """工具执行器：安全、可观测、可限流"""

    def __init__(self, llm_service, config):
        self.llm = llm_service
        self.config = config
        self.tools = self._register_tools()

    def _register_tools(self) -> dict:
        """注册所有可用工具"""
        return {
            "query_hydro_data": {
                "func": self._query_hydro_data,
                "description": "查询水文历史数据",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "station": {
                            "type": "string",
                            "description": "测站名称，如：宜昌站",
                            "enum": ["宜昌站", "汉口站", "大通站"],  # 限制可选值
                        },
                        "start_date": {
                            "type": "string",
                            "description": "开始日期，格式：YYYY-MM-DD",
                            "pattern": r"^\d{4}-\d{2}-\d{2}$",
                        },
                        "end_date": {"type": "string", "description": "结束日期"},
                    },
                    "required": ["station", "start_date"],
                },
                "rate_limit": "10/minute",  # 限流
                "timeout": 30,  # 超时
            },
            # ... 更多工具
        }

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict,
        user_id: str,
    ) -> ToolResult:
        """
        安全执行工具
        """
        # 1. 工具存在性检查
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                error=f"Tool {tool_name} not found",
                data=None,
            )

        tool = self.tools[tool_name]

        # 2. 参数校验
        try:
            self._validate_arguments(tool_name, arguments)
        except ValueError as e:
            return ToolResult(success=False, error=str(e), data=None)

        # 3. 权限检查（工具级）
        if not await self._check_permission(user_id, tool_name):
            return ToolResult(
                success=False,
                error="Permission denied",
                data=None,
            )

        # 4. 限流检查
        if not await self._check_rate_limit(user_id, tool_name):
            return ToolResult(
                success=False,
                error="Rate limit exceeded",
                data=None,
            )

        # 5. 执行（带超时）
        import asyncio
        try:
            result = await asyncio.wait_for(
                tool["func"](**arguments),
                timeout=tool.get("timeout", 30),
            )

            # 记录工具调用
            await self._log_tool_call(user_id, tool_name, arguments, result)

            return ToolResult(success=True, data=result, error=None)

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Tool {tool_name} timeout",
                data=None,
            )
        except Exception as e:
            logger.exception(f"Tool {tool_name} error: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                data=None,
            )

    async def run_agent_loop(
        self,
        user_message: str,
        user_id: str,
        max_steps: int = 10,
    ) -> dict:
        """
        Agent 主循环
        """
        messages = [
            {"role": "system", "content": "你是水利专业助手..."},
            {"role": "user", "content": user_message},
        ]

        for step in range(max_steps):
            # 1. 调用 LLM
            response = await self.llm.chat(
                messages=messages,
                tools=self._format_tools_for_openai(),
            )

            # 2. 检查是否需要调用工具
            if not response.choices[0].message.tool_calls:
                # 没有 tool_calls，说明完成
                break

            # 3. 执行工具
            tool_calls = response.choices[0].message.tool_calls
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                result = await self.execute_tool(tool_name, arguments, user_id)

                # 把结果加回 messages
                messages.append({
                    "role": "assistant",
                    "tool_calls": [tool_call],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

        return {"messages": messages, "steps": step + 1}
```

### 2.4.2 Agent 状态管理与恢复

```python
from typing import TypedDict
from datetime import datetime

class AgentState(TypedDict):
    session_id: str
    user_id: str
    messages: List[dict]
    current_step: int
    status: str  # running/complete/error
    created_at: datetime
    updated_at: datetime

class AgentStateManager:
    """Agent 状态管理：支持中断恢复"""

    def __init__(self, redis):
        self.redis = redis
        self.state_ttl = 3600  # 1小时过期

    async def save_state(self, state: AgentState):
        """保存 Agent 状态"""
        key = f"agent:state:{state['session_id']}"
        await self.redis.setex(
            key,
            self.state_ttl,
            json.dumps(state, default=str),
        )

    async def load_state(self, session_id: str) -> Optional[AgentState]:
        """加载 Agent 状态"""
        key = f"agent:state:{session_id}"
        data = await self.redis.get(key)
        if data:
            return json.loads(data)
        return None

    async def resume_agent(self, session_id: str):
        """恢复中断的 Agent"""
        state = await self.load_state(session_id)
        if not state or state["status"] != "running":
            raise ValueError(f"No running agent for session {session_id}")

        # 从上次中断处继续
        executor = ToolExecutor(...)
        return await executor.run_agent_loop(
            user_message="",
            user_id=state["user_id"],
            initial_messages=state["messages"],
            initial_step=state["current_step"],
        )
```

---

# 三、Web 全栈开发（生产级）

## 3.1 FastAPI 后端架构

### 3.1.1 项目结构规范

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI 应用入口
│   ├── config.py                  # 配置管理（Pydantic Settings）
│   ├── dependencies.py            # 依赖注入
│   │
│   ├── api/                       # 路由层
│   │   ├── __init__.py
│   │   ├── v1/                    # API 版本管理
│   │   │   ├── __init__.py
│   │   │   ├── chat.py            # /api/v1/chat
│   │   │   ├── rag.py             # /api/v1/rag
│   │   │   ├── agent.py           # /api/v1/agent
│   │   │   └── admin.py           # /api/v1/admin
│   │   └── deps.py                # 路由依赖（认证等）
│   │
│   ├── core/                      # 核心功能
│   │   ├── security.py            # JWT、密码哈希
│   │   ├── auth.py                # 认证逻辑
│   │   ├── rate_limit.py          # 限流
│   │   └── logger.py              # 日志配置
│   │
│   ├── models/                    # ORM 模型
│   │   ├── user.py
│   │   ├── conversation.py
│   │   └── document.py
│   │
│   ├── schemas/                   # Pydantic 模型（请求/响应）
│   │   ├── chat.py
│   │   ├── user.py
│   │   └── common.py
│   │
│   ├── services/                  # 业务逻辑
│   │   ├── llm_service.py
│   │   ├── rag_service.py
│   │   ├── agent_service.py
│   │   └── auth_service.py
│   │
│   └── utils/                     # 工具函数
│       ├── retry.py
│       ├── validators.py
│       └── formatters.py
│
├── tests/                         # 测试
│   ├── api/
│   ├── services/
│   └── conftest.py
│
├── scripts/                       # 离线脚本
│   ├── build_index.py             # 构建 RAG 索引
│   └── migrate_data.py
│
├── alembic/                       # 数据库迁移
│   └── versions/
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx.conf
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

### 3.1.2 配置管理

```python
# app/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """配置管理：环境变量优先，.env 文件补充"""

    # 应用配置
    APP_NAME: str = "Water AI Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"  # development/staging/production

    # 服务配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    WORKERS: int = 4

    # 数据库
    DATABASE_URL: str
    REDIS_URL: str

    # LLM 配置
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: Optional[str] = None
    DEFAULT_MODEL: str = "gpt-4o"
    FALLBACK_MODEL: str = "deepseek-chat"

    # RAG 配置
    VECTOR_DB_URL: str
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # 安全配置
    SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7天

    # 限流配置
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10

    # 监控配置
    SENTRY_DSN: Optional[str] = None
    LOG_LEVEL: str = "INFO"

    # OSS 配置
    OSS_ENDPOINT: str
    OSS_ACCESS_KEY: str
    OSS_SECRET_KEY: str
    OSS_BUCKET: str

    class Config:
        env_file = ".env"
        case_sensitive = True

# 单例
settings = Settings()
```

### 3.1.3 统一异常处理与响应

```python
# app/core/exceptions.py
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from typing import Union

class AppException(Exception):
    """应用基础异常"""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code

# 业务异常
class NotFoundException(AppException):
    def __init__(self, message: str = "Resource not found"):
        super().__init__(
            message=message,
            code="NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
        )

class BadRequestException(AppException):
    def __init__(self, message: str = "Bad request"):
        super().__init__(
            message=message,
            code="BAD_REQUEST",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

class UnauthorizedException(AppException):
    def __init__(self, message: str = "Unauthorized"):
        super().__init__(
            message=message,
            code="UNAUTHORIZED",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

class RateLimitException(AppException):
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message=message,
            code="RATE_LIMIT",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )

class LLMException(AppException):
    """LLM 调用异常"""
    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="LLM_ERROR",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

# 统一响应格式
class APIResponse:
    @staticmethod
    def success(data: any = None, message: str = "Success"):
        return {
            "success": True,
            "code": "SUCCESS",
            "message": message,
            "data": data,
        }

    @staticmethod
    def error(message: str, code: str = "ERROR", details: any = None):
        return {
            "success": False,
            "code": code,
            "message": message,
            "details": details,
        }

# 全局异常处理器
def setup_exception_handlers(app: FastAPI):
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        return JSONResponse(
            status_code=exc.status_code,
            content=APIResponse.error(
                message=exc.message,
                code=exc.code,
            ),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIResponse.error(
                message="Internal server error",
                code="INTERNAL_ERROR",
            ) if not settings.DEBUG else APIResponse.error(
                message=str(exc),
                code="INTERNAL_ERROR",
            ),
        )
```

### 3.1.4 限流实现

```python
# app/core/rate_limit.py
from fastapi import Request, HTTPException
from redis import Redis
import asyncio

class RateLimiter:
    """基于 Redis 的滑动窗口限流"""

    def __init__(self, redis: Redis):
        self.redis = redis

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window: int,  # 秒
    ) -> bool:
        """
        滑动窗口算法
        """
        now = asyncio.get_event_loop().time()
        window_start = now - window

        pipe = self.redis.pipeline()

        # 1. 移除窗口外的记录
        pipe.zremrangebyscore(key, 0, window_start)

        # 2. 计数
        pipe.zcard(key)

        # 3. 添加当前请求
        pipe.zadd(key, {str(now): now})

        # 4. 设置过期
        pipe.expire(key, window + 1)

        results = await pipe.execute()

        count = results[1]
        return count < limit

# 依赖注入使用
async def rate_limit_dependency(
    request: Request,
    user_id: str = Depends(get_current_user),
):
    limiter = RateLimiter(redis_client)

    key = f"rate_limit:user:{user_id}"
    allowed = await limiter.is_allowed(
        key=key,
        limit=settings.RATE_LIMIT_PER_MINUTE,
        window=60,
    )

    if not allowed:
        raise RateLimitException("Too many requests")

# 使用示例
@app.post("/api/chat", dependencies=[Depends(rate_limit_dependency)])
async def chat_endpoint(request: ChatRequest):
    ...
```

## 3.2 数据库工程

### 3.2.1 异步 ORM 实践

```python
# app/database.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from contextlib import asynccontextmanager

class Base(DeclarativeBase):
    pass

class Database:
    def __init__(self, url: str):
        self.engine = create_async_engine(
            url,
            echo=settings.DEBUG,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,  # 连接健康检查
        )
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @asynccontextmanager
    async def session(self):
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

db = Database(settings.DATABASE_URL)

# 使用
async def get_user(user_id: int) -> Optional[User]:
    async with db.session() as session:
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
```

### 3.2.2 对话历史存储优化

```python
# app/models/conversation.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255))
    model = Column(String(50))  # 使用的模型
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user/assistant/system
    content = Column(Text, nullable=False)

    # 扩展字段（用于存储 LLM 元信息）
    tokens = Column(Integer)
    latency_ms = Column(Integer)
    model = Column(String(50))

    # RAG 相关
    retrieved_docs = Column(JSON)  # 存储检索到的文档 ID

    created_at = Column(DateTime, server_default=func.now())

    conversation = relationship("Conversation", back_populates="messages")

# 查询优化：避免 N+1
async def get_conversation_with_messages(conversation_id: int) -> Optional[Conversation]:
    async with db.session() as session:
        result = await session.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .options(selectinload(Conversation.messages))  # 预加载
        )
        return result.scalar_one_or_none()
```

### 3.2.3 缓存策略

```python
# app/core/cache.py
from typing import Optional, Callable
from functools import wraps
import hashlib
import json

class CacheService:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def get(self, key: str) -> Optional[str]:
        return await self.redis.get(key)

    async def set(
        self,
        key: str,
        value: str,
        expire: int = 3600,
    ):
        await self.redis.setex(key, expire, value)

    async def delete(self, pattern: str):
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)

    def cached(
        self,
        expire: int = 3600,
        key_prefix: str = "",
    ):
        """缓存装饰器"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 生成缓存 key
                key_data = f"{key_prefix}:{args}:{kwargs}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()

                # 尝试从缓存获取
                cached = await self.get(cache_key)
                if cached:
                    return json.loads(cached)

                # 执行函数
                result = await func(*args, **kwargs)

                # 存入缓存
                await self.set(
                    cache_key,
                    json.dumps(result, ensure_ascii=False),
                    expire,
                )

                return result
            return wrapper
        return decorator

# 使用示例
cache = CacheService(redis_client)

@cache.cached(expire=1800, key_prefix="rag:retrieve")
async def retrieve_with_cache(query: str, filters: dict = None):
    # RAG 检索逻辑
    ...

# LLM 响应缓存（针对相同问题）
async def llm_chat_with_cache(messages: list, model: str):
    # 生成缓存 key（忽略 system prompt 的细微差异）
    content_key = json.dumps([m["content"] for m in messages if m["role"] != "system"])
    cache_key = f"llm:cache:{model}:{hashlib.md5(content_key.encode()).hexdigest()}"

    cached = await cache.get(cache_key)
    if cached:
        logger.info("LLM cache hit")
        return json.loads(cached)

    response = await llm_service.chat(messages, model)

    await cache.set(
        cache_key,
        json.dumps(response.choices[0].message.content),
        expire=86400,  # 24小时
    )

    return response
```

## 3.3 前端工程（够用就行）

### 3.3.1 React 聊天组件（完整版）

```jsx
// components/ChatPanel.jsx
import { useState, useRef, useEffect, useCallback } from 'react';
import { useSessionStorage } from '../hooks/useSessionStorage';

export function ChatPanel({ apiBase = '/api/v1' }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const abortControllerRef = useRef(null);
  const messagesEndRef = useRef(null);

  // 自动滚动到底部
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // 发送消息
  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    setError(null);

    // 创建可中断的请求
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(`${apiBase}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [...messages, userMsg].map(m => ({
            role: m.role,
            content: m.content,
          })),
          stream: true,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      // 创建助手消息
      const assistantMsg = { role: 'assistant', content: '' };
      setMessages(prev => [...prev, assistantMsg]);

      // 读取流
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // 保留未完成的行

        for (const line of lines) {
          if (!line.trim() || !line.startsWith('data: ')) continue;

          const data = line.slice(6).trim();
          if (data === '[DONE]') break;

          try {
            const parsed = JSON.parse(data);
            if (parsed.type === 'content') {
              assistantMsg.content += parsed.text;
              // 实时更新
              setMessages(prev => [
                ...prev.slice(0, -1),
                { ...assistantMsg },
              ]);
            } else if (parsed.type === 'error') {
              throw new Error(parsed.message);
            }
          } catch (e) {
            console.error('Parse error:', e);
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('Request aborted');
      } else {
        setError(err.message);
        setMessages(prev => [...prev.slice(0, -1)]); // 移除失败的助手消息
      }
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  };

  // 停止生成
  const stopGeneration = () => {
    abortControllerRef.current?.abort();
    setLoading(false);
  };

  // 重新生成
  const regenerate = async () => {
    const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
    if (lastUserMsg) {
      setMessages(prev => prev.slice(0, -1)); // 移除最后的助手消息
      setInput(lastUserMsg.content);
      await sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* 错误提示 */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-2">
          {error}
        </div>
      )}

      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-2xl px-4 py-2 rounded-lg ${
                msg.role === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white border shadow-sm'
              }`}
            >
              <div className="whitespace-pre-wrap">{msg.content}</div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* 输入框 */}
      <div className="border-t bg-white p-4">
        <div className="flex items-center space-x-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder="输入问题... (Shift+Enter 换行)"
            className="flex-1 border rounded-lg px-4 py-2 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows={2}
            disabled={loading}
          />
          {loading ? (
            <button
              onClick={stopGeneration}
              className="px-6 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600"
            >
              停止
            </button>
          ) : (
            <button
              onClick={sendMessage}
              disabled={!input.trim()}
              className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300"
            >
              发送
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
```

### 3.3.2 自定义 Hooks

```jsx
// hooks/useSessionStorage.js
import { useState, useEffect } from 'react';

export function useSessionStorage(key, initialValue) {
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.sessionStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      return initialValue;
    }
  });

  const setValue = (value) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.sessionStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(error);
    }
  };

  return [storedValue, setValue];
}

// hooks/useDebounce.js
import { useState, useEffect } from 'react';

export function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}
```

---

# 四、部署与运维（生产就绪）

## 4.1 Docker 容器化最佳实践

### 4.1.1 多阶段构建 Dockerfile

```dockerfile
# docker/Dockerfile
# 多阶段构建：减少镜像大小，提高安全性

# ============ 阶段1：构建 ============
FROM python:3.11-slim as builder

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /build

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖到临时目录
RUN pip install --user --no-cache-dir -r requirements.txt

# ============ 阶段2：运行 ============
FROM python:3.11-slim

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 设置工作目录
WORKDIR /app

# 从构建阶段复制依赖
COPY --from=builder /root/.local /root/.local

# 确保 Python 能找到已安装的包
ENV PATH=/root/.local/bin:$PATH

# 复制应用代码
COPY ./app ./app
COPY ./alembic ./alembic
COPY ./alembic.ini .

# 创建必要的目录
RUN mkdir -p /app/logs /app/tmp && \
    chown -R appuser:appuser /app

# 切换到非 root 用户
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**为什么多阶段构建？**
- 最终镜像只包含运行时依赖，不包含构建工具（gcc、g++）
- 镜像大小从 1GB+ 减少到 200MB 左右
- 减少攻击面（构建工具可能有漏洞）

### 4.1.2 镜像大小优化技巧

```dockerfile
# 优化技巧

# 1. 合并 RUN 指令（减少层数）
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*  # 清理缓存

# 2. 利用 Docker 缓存（变化少的放前面）
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .  # 代码经常变，放后面

# 3. 使用 .dockerignore
# .dockerignore
__pycache__
*.pyc
*.pyo
*.pyd
.git
.gitignore
.env
.venv
venv/
tests/
*.md

# 4. 选择合适的基础镜像
# python:3.11-slim   ~ 100MB（推荐）
# python:3.11-alpine ~ 50MB（最小，但有兼容性问题）
# python:3.11       ~ 900MB（完整，不推荐生产）
```

### 4.1.3 Docker Compose 完整配置

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  # ============ FastAPI 应用 ============
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ../app:/app/app  # 开发模式：热重载
      - app-logs:/app/logs
    restart: unless-stopped
    networks:
      - app-network
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M

  # ============ Nginx 反向代理 ============
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ../docker/nginx.conf:/etc/nginx/nginx.conf:ro
      - ../docker/ssl:/etc/nginx/ssl:ro
      - static-files:/static
    depends_on:
      - api
    restart: unless-stopped
    networks:
      - app-network

  # ============ PostgreSQL ============
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - app-network

  # ============ Redis ============
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - app-network

  # ============ Celery Worker (异步任务) ============
  celery-worker:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    command: celery -A app.tasks.celery_app worker --loglevel=info --concurrency=4
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - postgres
    volumes:
      - ../app:/app/app
    restart: unless-stopped
    networks:
      - app-network

  # ============ Celery Beat (定时任务) ============
  celery-beat:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    command: celery -A app.tasks.celery_app beat --loglevel=info
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    volumes:
      - ../app:/app/app
    restart: unless-stopped
    networks:
      - app-network

  # ============ Prometheus (监控) ============
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ../docker/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped
    networks:
      - app-network

  # ============ Grafana (可视化) ============
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ../docker/grafana-dashboards:/etc/grafana/provisioning/dashboards:ro
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - app-network

volumes:
  postgres-data:
  redis-data:
  app-logs:
  static-files:
  prometheus-data:
  grafana-data:

networks:
  app-network:
    driver: bridge
```

### 4.1.4 生产部署 Checklist

```yaml
# 部署前检查清单
deployment_checklist:

  安全检查:
    - 确认无硬编码密钥
    - 确认 .env 文件不提交到 Git
    - 确认使用非 root 用户运行容器
    - 确认敏感数据已加密存储

  配置检查:
    - 数据库连接池大小合理
    - Redis 连接数限制
    - 日志级别设置为 INFO 或 WARNING
    - 时区设置正确 (TZ=Asia/Shanghai)

  资源检查:
    - 容器内存限制设置
    - CPU 限制设置
    - 磁盘空间充足 (至少 20% 余量)
    - 数据库备份空间

  监控检查:
    - Prometheus 采集配置正确
    - 告警规则已配置
    - 日志采集已配置
    - 健康检查端点可用

  高可用检查:
    - 多副本部署 (replicas >= 2)
    - 数据库主从/集群配置
    - Redis 持久化开启
    - Nginx 负载均衡配置
```

## 4.2 Nginx 反向代理配置

```nginx
# docker/nginx.conf

user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # 日志格式
    log_format main '$remote_addr - $remote_user [$time_local] '
                    '"$request" $status $body_bytes_sent '
                    '"$http_referer" "$http_user_agent" '
                    '$request_time $upstream_response_time';

    access_log /var/log/nginx/access.log main;

    # 性能优化
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 100M;

    # Gzip 压缩
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml application/json application/javascript;

    # 限流配置
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_conn_zone $binary_remote_addr zone=conn_limit:10m;

    # 上游服务器
    upstream api_backend {
        least_conn;  # 最少连接负载均衡
        server api:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }

    server {
        listen 80;
        server_name example.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name example.com;

        # SSL 证书
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;

        # 静态文件
        location /static/ {
            alias /static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }

        # API 路由
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            limit_conn conn_limit 10;

            proxy_pass http://api_backend;
            proxy_http_version 1.1;

            # 请求头
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # 超时配置（重要！LLM 请求可能很慢）
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;

            # SSE 支持
            proxy_buffering off;
            proxy_cache off;
            proxy_set_header Connection '';
            proxy_set_header X-Accel-Buffering no;
            chunked_transfer_encoding on;
        }

        # 健康检查
        location /health {
            access_log off;
            proxy_pass http://api_backend/health;
        }
    }
}
```

## 4.3 监控与告警完整方案

### 4.3.1 应用层监控

```python
# app/monitoring.py
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps

# ============ Prometheus 指标定义 ============

# 请求计数
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
)

# 请求延迟
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
)

# LLM 调用
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM API calls',
    ['model', 'status'],
)

llm_request_duration_seconds = Histogram(
    'llm_request_duration_seconds',
    'LLM request latency',
    ['model'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total LLM tokens consumed',
    ['model', 'type'],  # type: prompt/completion
)

# RAG 指标
rag_retrieval_duration_seconds = Histogram(
    'rag_retrieval_duration_seconds',
    'RAG retrieval latency',
    ['method'],  # vector/bm25/hybrid
)

rag_retrieved_docs_count = Histogram(
    'rag_retrieved_docs_count',
    'Number of documents retrieved',
    buckets=[1, 3, 5, 10, 20, 50],
)

# 业务指标
active_conversations = Gauge(
    'active_conversations',
    'Number of active conversations',
)

daily_users = Gauge(
    'daily_users',
    'Number of daily active users',
)

# 应用信息
app_info = Info(
    'app',
    'Application information',
)

# ============ 中间件 ============
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # 处理请求
        response = await call_next(request)

        # 记录指标
        duration = time.time() - start_time
        method = request.method
        path = request.url.path
        status = response.status_code

        http_requests_total.labels(
            method=method,
            endpoint=path,
            status=status,
        ).inc()

        http_request_duration_seconds.labels(
            method=method,
            endpoint=path,
        ).observe(duration)

        return response

# ============ 装饰器 ============
def monitor_llm_call(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        model = kwargs.get('model', 'unknown')
        start = time.time()

        try:
            result = await func(*args, **kwargs)

            # 记录成功
            llm_requests_total.labels(model=model, status='success').inc()
            llm_request_duration_seconds.labels(model=model).observe(time.time() - start)

            # 记录 token
            if hasattr(result, 'usage'):
                llm_tokens_total.labels(model=model, type='prompt').inc(result.usage.prompt_tokens)
                llm_tokens_total.labels(model=model, type='completion').inc(result.usage.completion_tokens)

            return result

        except Exception as e:
            llm_requests_total.labels(model=model, status='error').inc()
            raise

    return wrapper
```

### 4.3.2 Prometheus 配置

```yaml
# docker/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: []

rule_files:
  - '/etc/prometheus/alerts/*.yml'

scrape_configs:
  # FastAPI 应用
  - job_name: 'fastapi'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # PostgreSQL
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:9187']

  # Redis
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']

  # Node Exporter (系统指标)
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

### 4.3.3 告警规则

```yaml
# docker/alerts/api.yml
groups:
  - name: api_alerts
    interval: 30s
    rules:
      # 高错误率
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) /
          sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # 高延迟
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
          ) > 3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High P95 latency"
          description: "P95 latency is {{ $value }}s"

      # LLM 调用失败率高
      - alert: LLMHighFailureRate
        expr: |
          sum(rate(llm_requests_total{status="error"}[5m])) /
          sum(rate(llm_requests_total[5m])) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "LLM API failure rate high"
          description: "LLM failure rate is {{ $value | humanizePercentage }}"

      # 服务不可用
      - alert: ServiceDown
        expr: up{job="fastapi"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "{{ $labels.instance }} is not responding"

  - name: business_alerts
    interval: 1m
    rules:
      # 日活异常下降
      - alert: LowDailyUsers
        expr: daily_users < 10
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Daily active users unusually low"

      # Token 消耗异常
      - alert: HighTokenConsumption
        expr: |
          sum(rate(llm_tokens_total[1h])) > 10000
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Unusually high token consumption"
```

### 4.3.4 结构化日志

```python
# app/core/logging.py
import logging
import sys
from datetime import datetime
import json
from contextvars import ContextVar

# 请求上下文
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')

class JSONFormatter(logging.Formatter):
    """结构化日志输出"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'request_id': request_id_var.get(),
            'user_id': user_id_var.get(),
        }

        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # 添加额外字段
        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        return json.dumps(log_data, ensure_ascii=False)

def setup_logging():
    """配置日志系统"""
    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)

    # 第三方库日志级别
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)

    return root_logger

# 使用示例
logger = logging.getLogger(__name__)

# 在中间件中设置 request_id
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    import uuid
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)

    logger.info("request_started", extra={
        'method': request.method,
        'path': request.url.path,
    })

    response = await call_next(request)

    logger.info("request_completed", extra={
        'status': response.status_code,
    })

    return response
```

---

# 五、故障排查实战

## 5.1 常见故障诊断表

| 故障现象 | 可能原因 | 排查步骤 | 解决方案 |
|---------|---------|---------|---------|
| 502 Bad Gateway | 后端服务挂了 | 1. `docker ps` 检查容器状态<br>2. 查看后端日志 | 重启服务，检查错误日志 |
| 503 Service Unavailable | 后端过载/限流 | 1. 检查 CPU/内存<br>2. 查看限流日志 | 扩容、优化代码、调整限流 |
| 504 Gateway Timeout | 后端响应慢 | 1. 查看 LLM 调用延迟<br>2. 检查数据库查询 | 增加超时时间、优化慢查询 |
| 连接数据库失败 | 网络问题/认证失败 | 1. `ping postgres`<br>2. 检查连接字符串 | 修复网络、检查密码 |
| Redis 连接超时 | Redis 过载/网络问题 | 1. `redis-cli ping`<br>2. 检查 Redis 日志 | 检查 Redis 内存、优化配置 |
| LLM 调用失败 | API 密钥过期/额度不足 | 1. 检查 API 密钥<br>2. 查看错误消息 | 更新密钥、切换备用模型 |
| 内存溢出 OOM | 内存泄漏/请求过多 | 1. `docker stats`<br>2. 检查缓存配置 | 限制并发、优化内存使用 |
| CPU 100% | 死循环/计算密集 | 1. `top` 查看进程<br>2. 分析 CPU profile | 优化算法、异步化 |

## 5.2 故障排查流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        故障发生：用户报告问题                             │
└──────────────────────────────────────┬──────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        Step 1: 确认问题范围                              │
│  • 单个用户还是全部用户？                                                │
│  • 特定功能还是全部功能？                                                │
│  • 特定时间段还是持续发生？                                              │
└──────────────────────────────────────┬──────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        Step 2: 检查服务状态                              │
│  • `docker ps` - 容器是否运行？                                         │
│  • `docker logs <container>` - 有无错误日志？                            │
│  • `docker stats` - 资源使用情况？                                       │
└──────────────────────────────────────┬──────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        Step 3: 查看监控指标                              │
│  • Prometheus - 错误率、延迟、QPS 是否异常？                             │
│  • Grafana - 哪个时间点开始异常？                                        │
│  • 告警 - 有没有触发告警？                                               │
└──────────────────────────────────────┬──────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        Step 4: 定位具体问题                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │  应用层问题  │  │ 数据库问题  │  │  LLM API   │  │ 网络问题   │   │
│  │  查看代码   │  │  慢查询    │  │  密钥/额度  │  │  ping测试  │   │
│  │  日志       │  │  连接池    │  │  超时设置  │  │  防火墙    │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└──────────────────────────────────────┬──────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        Step 5: 实施修复                                  │
│  • 快速恢复：重启服务、切换备用模型、降级功能                            │
│  • 根本修复：修复 bug、优化查询、扩容                                    │
└──────────────────────────────────────┬──────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        Step 6: 验证与复盘                                │
│  • 验证修复是否有效                                                     │
│  • 记录故障报告                                                         │
│  • 更新监控/告警规则                                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

## 5.3 排查命令速查

```bash
# ========== 服务状态 ==========
# 检查所有容器状态
docker ps -a

# 查看容器日志（最后100行）
docker logs --tail 100 <container>

# 实时查看日志
docker logs -f <container>

# 查看资源使用
docker stats

# 进入容器调试
docker exec -it <container> /bin/bash

# ========== 应用调试 ==========
# 检查健康状态
curl http://localhost:8000/health

# 查看 Prometheus 指标
curl http://localhost:8000/metrics

# 测试 API 端点
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'

# ========== 数据库 ==========
# 连接 PostgreSQL
docker exec -it postgres psql -U postgres -d water_ai

# 查看活跃连接
SELECT * FROM pg_stat_activity WHERE datname = 'water_ai';

# 查看慢查询
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC LIMIT 10;

# 查看表大小
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;

# ========== Redis ==========
# 连接 Redis
docker exec -it redis redis-cli

# 查看内存使用
INFO memory

# 查看连接数
CLIENT LIST

# 查看所有 key
KEYS *

# 清空特定前缀的 key
EVAL "return redis.call('del', unpack(redis.call('keys', ARGV[1])))" 0 cache:*

# ========== 网络 ==========
# 测试网络连通性
docker exec api ping postgres

# 查看端口监听
netstat -tlnp

# 抓包分析
tcpdump -i any port 8000 -n

# ========== 性能分析 ==========
# CPU 分析
top -p $(pgrep -f uvicorn)

# 内存分析
pmap $(pgrep -f uvicorn)

# Python profile
python -m cProfile -o profile.stats app/main.py
```

## 5.4 典型故障案例

### 案例1：OOM 导致服务频繁重启

**现象**：
- 服务频繁重启
- 日志显示 `MemoryError` 或直接无日志退出

**排查**：
```bash
# 1. 查看容器重启次数
docker ps -a  # 显示 Restart 值

# 2. 查看内存限制和使用
docker stats

# 3. 分析内存占用
docker exec api python -c "
import psutil
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
import gc
gc.collect()
print('After GC:', process.memory_info().rss / 1024 / 1024, 'MB')
"
```

**根因**：
- LLM 响应缓存未设置过期，Redis 内存持续增长
- 向量索引加载到内存，多个 worker 重复加载

**解决**：
```python
# 1. 修复缓存过期
await redis.setex(key, 3600, value)  # 1小时过期

# 2. 共享向量索引（单例模式）
class VectorStoreSingleton:
    _instance = None
    _store = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def store(self):
        if self._store is None:
            self._store = self._load_store()
        return self._store
```

### 案例2：LLM 调用间歇性超时

**现象**：
- 部分请求返回 504 超时
- 日志显示 `APITimeoutError`

**排查**：
```python
# 添加详细日志
async def chat_with_logging(messages):
    start = time.time()
    try:
        response = await llm.chat(messages)
        logger.info(f"LLM success in {time.time()-start:.2f}s")
        return response
    except openai.APITimeoutError:
        logger.warning(f"LLM timeout after {time.time()-start:.2f}s")
        # 尝试备用模型
        return await llm.chat(messages, model=settings.FALLBACK_MODEL)
```

**根因**：
- OpenAI API 确实有延迟波动
- 超时设置太短（10s）

**解决**：
```python
# 1. 增加超时时间
client = openai.AsyncOpenAI(timeout=60.0)

# 2. 实现重试机制
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def chat(...)

# 3. 前端显示"AI 正在思考..."，给用户预期
```

### 案例3：数据库连接池耗尽

**现象**：
- 新请求阻塞
- 日志显示 `connection pool exhausted`

**排查**：
```sql
-- 查看 PostgreSQL 连接数
SELECT count(*) FROM pg_stat_activity;
SELECT max_conn FROM pg_settings WHERE name = 'max_connections';

-- 查看谁占用了连接
SELECT pid, usename, state, query
FROM pg_stat_activity
WHERE datname = 'water_ai'
ORDER BY state_change DESC;
```

**根因**：
- 连接未释放（未使用 `async with`）
- 并发请求 > 连接池大小

**解决**：
```python
# 1. 确保连接释放
async with db.session() as session:
    ...  # 自动释放

# 2. 调整连接池大小
engine = create_async_engine(
    url,
    pool_size=20,      # 增加连接池
    max_overflow=10,   # 允许溢出
    pool_timeout=30,   # 获取连接超时
)

# 3. 慢查询优化
# 添加索引
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
```

---

# 六、面试高频问题速查

## LLM 相关

| 问题 | 核心要点 |
|------|----------|
| **RAG 召回质量差怎么优化？** | 1) Chunk 策略调整（500-1000 token，有 overlap）<br>2) 换 Embedding 模型（bge-m3 中文更好）<br>3) 混合检索（向量 + BM25）<br>4) 加 Rerank（CrossEncoder）<br>5) 元数据过滤 |
| **如何防止 LLM 幻觉？** | 1) Prompt 明确约束"不确定说不知道"<br>2) 强制返回 sources 字段<br>3) LLM-as-judge 事后验证 faithfulness<br>4) 降低 temperature（0~0.3）<br>5) 检索质量提升 |
| **Agent 失控怎么办？** | 1) 最大步数限制<br>2) 危险操作人工确认<br>3) 工具参数校验<br>4) 完整 trace 日志<br>5) 敏感操作权限控制 |
| **流式输出怎么实现？** | 1) 后端：`stream=True` + SSE（`text/event-stream`）<br>2) 前端：`ReadableStream` 读取<br>3) Nginx 关闭缓冲：`proxy_buffering off` |
| **LLM 成本怎么控制？** | 1) 缓存相同问题响应<br>2) 简单任务用小模型（gpt-4o-mini）<br>3) Prompt 压缩（去掉冗余）<br>4) Token 计费监控<br>5) 降级到开源模型 |

## Web 开发相关

| 问题 | 核心要点 |
|------|----------|
| **FastAPI 为什么适合 LLM 应用？** | 1) 原生异步，不阻塞<br>2) 自动 OpenAPI 文档<br>3) Pydantic 校验（结构化输出）<br>4) 流式响应支持好 |
| **怎么设计 RESTful API？** | 1) 资源命名：名词，不用动词<br>2) HTTP 方法语义：GET/POST/PUT/DELETE<br>3) 版本管理：`/api/v1/`<br>4) 统一响应格式：`{success, code, message, data}`<br>5) 错误码规范 |
| **如何处理并发请求？** | 1) 异步 I/O（async/await）<br>2) 连接池复用<br>3) 限流保护<br>4) 队列缓冲（Celery）<br>5) 水平扩展 |
| **数据库查询慢怎么办？** | 1) EXPLAIN 分析执行计划<br>2) 添加索引<br>3) 避免 N+1（用 join 或预加载）<br>4) 分页查询<br>5) 读写分离 |
| **Session 管理方案？** | 1) JWT 无状态（推荐）<br>2) Redis 存储 Session<br>3) Cookie 安全设置（HttpOnly, Secure）<br>4) CSRF 保护 |

## 系统设计相关

| 问题 | 核心要点 |
|------|----------|
| **设计一个对话系统** | 1) 前端：WebSocket/SSE 流式<br>2) 后端：FastAPI + 异步<br>3) 存储：PostgreSQL（历史）+ Redis（缓存）<br>4) LLM：带重试和降级<br>5) 监控：延迟、错误率、Token |
| **监控指标有哪些？** | 1) 基础：QPS、延迟（P50/P95）、错误率<br>2) LLM：Token 消耗、模型分布<br>3) RAG：召回质量、命中率<br>4) 业务：DAU、对话数、满意度 |
| **如何保证高可用？** | 1) 多副本部署（K8s/Docker Swarm）<br>2) 数据库主从/集群<br>3) Redis 持久化 + 哨兵<br>4) 限流 + 熔断<br>5) 优雅关闭 + 健康检查 |
| **私有化部署方案？** | 1) LLM：vLLM/Ollama 本地推理<br>2) 向量库：Milvus/pgvector<br>3) 整体：Docker Compose 一键部署<br>4) 监控：Prometheus + Grafana<br>5) 更新：蓝绿部署/金丝雀发布 |

## 数据结构/算法相关

| 问题 | 核心要点 |
|------|----------|
| **如何实现限流？** | 1) 滑动窗口：Redis ZSET<br>2) 令牌桶：恒定速率<br>3) 漏桶：平滑输出<br>4) 固定窗口：计数器（简单但不精准） |
| **如何去重？** | 1) 内存：HashSet（小数据）<br>2) 缓存：Redis SET/布隆过滤器<br>3) 数据库：唯一索引 |
| **相似度计算？** | 1) 文本：余弦相似度<br>2) 集合：Jaccard 相似度<br>3) 序列：编辑距离 |
| **负载均衡算法？** | 1) 轮询：简单<br>2) 最少连接：适合长请求<br>3) 一致性哈希：适合缓存<br>4) 加权：按性能分配 |

---

# 七、学习路径（3天执行版）

## Day 1：LLM 应用核心

- [ ] **上午**（4小时）
  - [ ] 跑通 OpenAI API 调用（streaming）
  - [ ] 实现 Prompt 模板系统
  - [ ] 用 instructor 实现结构化输出

- [ ] **下午**（4小时）
  - [ ] 用 LangChain 搭建 RAG（本地 PDF → ChromaDB）
  - [ ] 实现 Rerank
  - [ ] 写评估脚本（Hit Rate、MRR）

## Day 2：Web 全栈 + 部署

- [ ] **上午**（4小时）
  - [ ] FastAPI 写 `/api/chat` 接口（含流式）
  - [ ] PostgreSQL 存储对话历史
  - [ ] 实现认证中间件（JWT）

- [ ] **下午**（4小时）
  - [ ] React 搭聊天界面
  - [ ] Docker Compose 跑起来全套
  - [ ] Nginx 反向代理配置

## Day 3：工程化 + 面试准备

- [ ] **上午**（4小时）
  - [ ] 加监控（Prometheus + 结构化日志）
  - [ ] 写故障演练脚本
  - [ ] 完善 README 和架构图

- [ ] **下午**（4小时）
  - [ ] 复习高频面试题
  - [ ] 准备项目讲解（STAR 法则）
  - [ ] 模拟面试

---

*最后更新：2026-03-02*
