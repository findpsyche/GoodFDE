"""
FastAPI 服务封装示例

将本地模型封装为 REST API 服务，包括：
- 健康检查
- 限流
- 缓存
- 监控
- 日志
"""

import os
import time
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
import uvicorn

load_dotenv()

# 配置
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8080"))
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

# 创建 FastAPI 应用
app = FastAPI(
    title="Local LLM API",
    description="本地大语言模型 API 服务",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 简单的内存缓存
cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 3600  # 1小时

# 限流计数器
rate_limit_counter: Dict[str, list] = defaultdict(list)

# 请求统计
stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_tokens": 0,
    "cache_hits": 0
}


class GenerateRequest(BaseModel):
    """生成请求"""
    prompt: str = Field(..., description="输入提示")
    model: str = Field(DEFAULT_MODEL, description="模型名称")
    max_tokens: int = Field(512, description="最大生成 token 数")
    temperature: float = Field(0.7, description="温度参数", ge=0.0, le=2.0)
    top_p: float = Field(0.9, description="Top-p 采样", ge=0.0, le=1.0)
    stream: bool = Field(False, description="是否流式响应")


class GenerateResponse(BaseModel):
    """生成响应"""
    output: str
    model: str
    tokens: int
    elapsed_time: float
    cached: bool = False


def get_cache_key(request: GenerateRequest) -> str:
    """生成缓存键"""
    key_str = f"{request.model}:{request.prompt}:{request.max_tokens}:{request.temperature}:{request.top_p}"
    return hashlib.md5(key_str.encode()).hexdigest()


def check_rate_limit(client_ip: str) -> bool:
    """检查限流"""
    now = datetime.now()
    one_minute_ago = now - timedelta(minutes=1)

    # 清理过期的请求记录
    rate_limit_counter[client_ip] = [
        ts for ts in rate_limit_counter[client_ip]
        if ts > one_minute_ago
    ]

    # 检查是否超过限制
    if len(rate_limit_counter[client_ip]) >= RATE_LIMIT:
        return False

    # 记录本次请求
    rate_limit_counter[client_ip].append(now)
    return True


def get_from_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """从缓存获取"""
    if cache_key in cache:
        cached_data = cache[cache_key]
        # 检查是否过期
        if datetime.now() < cached_data["expires_at"]:
            stats["cache_hits"] += 1
            return cached_data["data"]
        else:
            # 删除过期缓存
            del cache[cache_key]
    return None


def save_to_cache(cache_key: str, data: Dict[str, Any]):
    """保存到缓存"""
    cache[cache_key] = {
        "data": data,
        "expires_at": datetime.now() + timedelta(seconds=CACHE_TTL)
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "Local LLM API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查 Ollama 服务
        response = requests.get(f"{OLLAMA_HOST}/", timeout=5)
        ollama_status = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        ollama_status = "unhealthy"

    return {
        "status": "healthy" if ollama_status == "healthy" else "degraded",
        "ollama": ollama_status,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    return {
        "stats": stats,
        "cache_size": len(cache),
        "rate_limit_clients": len(rate_limit_counter)
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, req: Request):
    """生成响应"""
    stats["total_requests"] += 1

    # 获取客户端 IP
    client_ip = req.client.host

    # 检查限流
    if not check_rate_limit(client_ip):
        stats["failed_requests"] += 1
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试"
        )

    # 检查缓存
    cache_key = get_cache_key(request)
    cached_result = get_from_cache(cache_key)

    if cached_result:
        return GenerateResponse(**cached_result, cached=True)

    # 调用 Ollama API
    try:
        data = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p
            }
        }

        start_time = time.time()
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=data,
            timeout=300
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            output = result.get("response", "")
            tokens = result.get("eval_count", 0)

            # 更新统计
            stats["successful_requests"] += 1
            stats["total_tokens"] += tokens

            # 构建响应
            response_data = {
                "output": output,
                "model": request.model,
                "tokens": tokens,
                "elapsed_time": elapsed
            }

            # 保存到缓存
            save_to_cache(cache_key, response_data)

            return GenerateResponse(**response_data)
        else:
            stats["failed_requests"] += 1
            raise HTTPException(
                status_code=response.status_code,
                detail="模型调用失败"
            )

    except requests.exceptions.Timeout:
        stats["failed_requests"] += 1
        raise HTTPException(status_code=504, detail="请求超时")
    except Exception as e:
        stats["failed_requests"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache")
async def clear_cache():
    """清除缓存"""
    cache.clear()
    return {"message": "缓存已清除", "cache_size": 0}


@app.get("/models")
async def list_models():
    """列出可用模型"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            return {
                "models": [
                    {
                        "name": m.get("name"),
                        "size": m.get("size"),
                        "modified": m.get("modified_at")
                    }
                    for m in models
                ]
            }
        else:
            raise HTTPException(status_code=500, detail="获取模型列表失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print(f"🚀 启动 API 服务: http://{API_HOST}:{API_PORT}")
    print(f"📚 API 文档: http://{API_HOST}:{API_PORT}/docs")
    print(f"🔧 Ollama 地址: {OLLAMA_HOST}")
    print(f"⚡ 限流: {RATE_LIMIT} 请求/分钟")

    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )
