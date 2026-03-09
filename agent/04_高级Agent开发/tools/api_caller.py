"""
API调用工具

功能：
- 调用REST API
- 处理认证
- 重试和错误处理

设计原则：
- 清晰的接口设计
- 完善的错误处理
- 详细的日志记录
"""

import requests
from typing import Dict, Optional, Any
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    重试装饰器

    Args:
        max_retries: 最大重试次数
        delay: 重试延迟（秒）
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"尝试 {attempt + 1}/{max_retries} 失败: {e}")
                    time.sleep(delay * (2 ** attempt))  # 指数退避
            return None
        return wrapper
    return decorator


class APICallerTool:
    """
    API调用工具

    使用场景：
    - 需要调用外部API
    - 需要处理认证和重试
    - 需要统一的错误处理

    注意事项：
    - 保护API密钥
    - 处理速率限制
    - 验证响应数据
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        """
        初始化API调用工具

        Args:
            base_url: API基础URL
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.headers = {}

    def set_auth(self, auth_type: str, credentials: Dict[str, str]):
        """
        设置认证信息

        Args:
            auth_type: 认证类型（'bearer', 'basic', 'api_key'）
            credentials: 认证凭据

        示例:
            >>> api = APICallerTool()
            >>> api.set_auth('bearer', {'token': 'your-token'})
        """
        if auth_type == 'bearer':
            self.headers['Authorization'] = f"Bearer {credentials['token']}"
        elif auth_type == 'basic':
            from requests.auth import HTTPBasicAuth
            self.session.auth = HTTPBasicAuth(
                credentials['username'],
                credentials['password']
            )
        elif auth_type == 'api_key':
            self.headers[credentials['header_name']] = credentials['api_key']

        logger.info(f"✅ 设置认证: {auth_type}")

    @retry_on_failure(max_retries=3)
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        发送GET请求

        Args:
            endpoint: API端点
            params: 查询参数

        Returns:
            API响应数据

        Raises:
            requests.RequestException: 请求失败

        示例:
            >>> api = APICallerTool(base_url="https://api.example.com")
            >>> result = api.get("/users", params={"page": 1})
        """
        url = self._build_url(endpoint)
        logger.info(f"🌐 GET {url}")

        response = self.session.get(
            url,
            params=params,
            headers=self.headers,
            timeout=self.timeout
        )

        return self._handle_response(response)

    @retry_on_failure(max_retries=3)
    def post(self, endpoint: str, data: Optional[Dict] = None, json: Optional[Dict] = None) -> Dict[str, Any]:
        """
        发送POST请求

        Args:
            endpoint: API端点
            data: 表单数据
            json: JSON数据

        Returns:
            API响应数据

        示例:
            >>> api = APICallerTool(base_url="https://api.example.com")
            >>> result = api.post("/users", json={"name": "Alice"})
        """
        url = self._build_url(endpoint)
        logger.info(f"🌐 POST {url}")

        response = self.session.post(
            url,
            data=data,
            json=json,
            headers=self.headers,
            timeout=self.timeout
        )

        return self._handle_response(response)

    @retry_on_failure(max_retries=3)
    def put(self, endpoint: str, data: Optional[Dict] = None, json: Optional[Dict] = None) -> Dict[str, Any]:
        """发送PUT请求"""
        url = self._build_url(endpoint)
        logger.info(f"🌐 PUT {url}")

        response = self.session.put(
            url,
            data=data,
            json=json,
            headers=self.headers,
            timeout=self.timeout
        )

        return self._handle_response(response)

    @retry_on_failure(max_retries=3)
    def delete(self, endpoint: str) -> Dict[str, Any]:
        """发送DELETE请求"""
        url = self._build_url(endpoint)
        logger.info(f"🌐 DELETE {url}")

        response = self.session.delete(
            url,
            headers=self.headers,
            timeout=self.timeout
        )

        return self._handle_response(response)

    def _build_url(self, endpoint: str) -> str:
        """构建完整URL"""
        if self.base_url:
            return f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        return endpoint

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        处理API响应

        Args:
            response: requests响应对象

        Returns:
            解析后的响应数据

        Raises:
            requests.HTTPError: HTTP错误
        """
        # 检查状态码
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            logger.error(f"❌ HTTP错误: {e}")
            logger.error(f"响应内容: {response.text[:500]}")
            raise

        # 解析JSON
        try:
            data = response.json()
            logger.info(f"✅ 请求成功: {response.status_code}")
            return {
                'success': True,
                'status_code': response.status_code,
                'data': data
            }
        except ValueError:
            # 非JSON响应
            return {
                'success': True,
                'status_code': response.status_code,
                'data': response.text
            }

    def validate_response(self, response: Dict[str, Any], required_fields: list) -> bool:
        """
        验证响应数据

        Args:
            response: API响应
            required_fields: 必需字段列表

        Returns:
            是否验证通过
        """
        if not response.get('success'):
            return False

        data = response.get('data', {})
        if not isinstance(data, dict):
            return False

        # 检查必需字段
        for field in required_fields:
            if field not in data:
                logger.warning(f"⚠️  缺少必需字段: {field}")
                return False

        logger.info("✅ 响应验证通过")
        return True


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    print("\n" + "="*50)
    print("API调用工具示例")
    print("="*50 + "\n")

    # 使用公开的测试API
    api = APICallerTool(base_url="https://jsonplaceholder.typicode.com")

    # 示例1：GET请求
    print("示例1: GET请求")
    try:
        result = api.get("/posts/1")
        print(f"成功: {result['success']}")
        print(f"标题: {result['data'].get('title', 'N/A')}")
    except Exception as e:
        print(f"失败: {e}")
    print()

    # 示例2：POST请求
    print("示例2: POST请求")
    try:
        result = api.post("/posts", json={
            "title": "Test Post",
            "body": "This is a test",
            "userId": 1
        })
        print(f"成功: {result['success']}")
        print(f"创建的ID: {result['data'].get('id', 'N/A')}")
    except Exception as e:
        print(f"失败: {e}")
    print()

    # 示例3：验证响应
    print("示例3: 验证响应")
    try:
        result = api.get("/posts/1")
        valid = api.validate_response(result, ['title', 'body', 'userId'])
        print(f"验证通过: {valid}")
    except Exception as e:
        print(f"失败: {e}")


if __name__ == "__main__":
    example_usage()
