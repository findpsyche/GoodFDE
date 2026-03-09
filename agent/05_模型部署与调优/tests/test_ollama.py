"""
测试 Ollama 部署
"""

import pytest
import requests
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


@pytest.fixture
def ollama_client():
    """Ollama 客户端 fixture"""
    return {
        "host": OLLAMA_HOST,
        "api_base": f"{OLLAMA_HOST}/api"
    }


def test_service_health(ollama_client):
    """测试服务健康状态"""
    response = requests.get(ollama_client["host"], timeout=5)
    assert response.status_code == 200


def test_list_models(ollama_client):
    """测试列出模型"""
    response = requests.get(f"{ollama_client['api_base']}/tags", timeout=10)
    assert response.status_code == 200

    data = response.json()
    assert "models" in data


def test_basic_inference(ollama_client):
    """测试基本推理"""
    # 先获取可用模型
    response = requests.get(f"{ollama_client['api_base']}/tags", timeout=10)
    models = response.json().get("models", [])

    if not models:
        pytest.skip("没有可用的模型")

    model = models[0].get("name")

    # 测试推理
    data = {
        "model": model,
        "prompt": "Say hello.",
        "stream": False,
        "options": {"num_predict": 10}
    }

    response = requests.post(
        f"{ollama_client['api_base']}/generate",
        json=data,
        timeout=60
    )

    assert response.status_code == 200

    result = response.json()
    assert "response" in result
    assert len(result["response"]) > 0


def test_streaming_inference(ollama_client):
    """测试流式推理"""
    response = requests.get(f"{ollama_client['api_base']}/tags", timeout=10)
    models = response.json().get("models", [])

    if not models:
        pytest.skip("没有可用的模型")

    model = models[0].get("name")

    data = {
        "model": model,
        "prompt": "Count to 3.",
        "stream": True,
        "options": {"num_predict": 10}
    }

    response = requests.post(
        f"{ollama_client['api_base']}/generate",
        json=data,
        stream=True,
        timeout=60
    )

    assert response.status_code == 200

    # 验证流式响应
    chunks = 0
    for line in response.iter_lines():
        if line:
            chunks += 1
            if chunks > 5:  # 只验证前几个chunk
                break

    assert chunks > 0


def test_invalid_model(ollama_client):
    """测试无效模型"""
    data = {
        "model": "nonexistent-model",
        "prompt": "Test",
        "stream": False
    }

    response = requests.post(
        f"{ollama_client['api_base']}/generate",
        json=data,
        timeout=10
    )

    # 应该返回错误
    assert response.status_code != 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
