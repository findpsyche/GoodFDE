"""
代码执行工具

功能：
- 安全执行Python代码
- 捕获输出和错误
- 限制执行时间和资源

设计原则：
- 安全第一：沙箱环境
- 清晰的错误信息
- 详细的执行日志
"""

import sys
import io
import contextlib
import signal
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """超时异常"""
    pass


def timeout_handler(signum, frame):
    """超时处理器"""
    raise TimeoutException("代码执行超时")


class CodeExecutorTool:
    """
    代码执行工具

    使用场景：
    - 需要执行Python代码
    - 需要验证代码逻辑
    - 需要测试代码片段

    注意事项：
    - 只在受信任的环境中使用
    - 限制执行时间
    - 禁止危险操作（文件系统、网络等）
    """

    def __init__(self, timeout: int = 5, max_output_length: int = 10000):
        """
        初始化代码执行工具

        Args:
            timeout: 执行超时时间（秒）
            max_output_length: 最大输出长度
        """
        self.timeout = timeout
        self.max_output_length = max_output_length

        # 禁止的模块和函数
        self.forbidden_imports = {
            'os', 'sys', 'subprocess', 'socket', 'urllib',
            'requests', 'shutil', 'pathlib'
        }

    def execute(self, code: str) -> Dict[str, Any]:
        """
        执行Python代码

        Args:
            code: Python代码字符串

        Returns:
            包含输出、错误、执行状态的字典

        示例:
            >>> executor = CodeExecutorTool()
            >>> result = executor.execute("print('Hello, World!')")
            >>> print(result['output'])
            'Hello, World!'
        """
        logger.info(f"🔧 执行代码: {code[:50]}...")

        # 安全检查
        if not self._is_safe_code(code):
            return {
                'success': False,
                'output': '',
                'error': '代码包含禁止的操作',
                'execution_time': 0
            }

        # 捕获输出
        stdout = io.StringIO()
        stderr = io.StringIO()

        result = {
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0
        }

        try:
            # 设置超时（仅Unix系统）
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)

            # 执行代码
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                # 创建受限的全局命名空间
                globals_dict = {
                    '__builtins__': {
                        'print': print,
                        'len': len,
                        'range': range,
                        'str': str,
                        'int': int,
                        'float': float,
                        'list': list,
                        'dict': dict,
                        'set': set,
                        'tuple': tuple,
                        'sum': sum,
                        'max': max,
                        'min': min,
                        'abs': abs,
                        'round': round,
                    }
                }

                exec(code, globals_dict)

            # 取消超时
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

            result['success'] = True
            result['output'] = stdout.getvalue()[:self.max_output_length]

            logger.info("✅ 代码执行成功")

        except TimeoutException:
            result['error'] = f"执行超时（>{self.timeout}秒）"
            logger.warning(f"⏰ 执行超时")

        except Exception as e:
            result['error'] = f"{type(e).__name__}: {str(e)}"
            result['output'] = stderr.getvalue()[:self.max_output_length]
            logger.error(f"❌ 执行失败: {e}")

        finally:
            # 确保取消超时
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

        return result

    def _is_safe_code(self, code: str) -> bool:
        """
        检查代码是否安全

        Args:
            code: 代码字符串

        Returns:
            是否安全
        """
        # 检查禁止的导入
        for forbidden in self.forbidden_imports:
            if f"import {forbidden}" in code or f"from {forbidden}" in code:
                logger.warning(f"⚠️  检测到禁止的导入: {forbidden}")
                return False

        # 检查危险函数
        dangerous_functions = ['eval', 'exec', 'compile', '__import__', 'open']
        for func in dangerous_functions:
            if func in code:
                logger.warning(f"⚠️  检测到危险函数: {func}")
                return False

        return True

    def validate_syntax(self, code: str) -> Dict[str, Any]:
        """
        验证代码语法

        Args:
            code: 代码字符串

        Returns:
            验证结果
        """
        try:
            compile(code, '<string>', 'exec')
            return {
                'valid': True,
                'error': None
            }
        except SyntaxError as e:
            return {
                'valid': False,
                'error': f"语法错误: {e.msg} (行 {e.lineno})"
            }


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    print("\n" + "="*50)
    print("代码执行工具示例")
    print("="*50 + "\n")

    executor = CodeExecutorTool()

    # 示例1：简单计算
    print("示例1: 简单计算")
    code1 = """
result = 2 + 3 * 4
print(f"结果: {result}")
"""
    result1 = executor.execute(code1)
    print(f"成功: {result1['success']}")
    print(f"输出: {result1['output']}")
    print()

    # 示例2：循环
    print("示例2: 循环")
    code2 = """
for i in range(5):
    print(f"数字: {i}")
"""
    result2 = executor.execute(code2)
    print(f"输出:\n{result2['output']}")
    print()

    # 示例3：错误处理
    print("示例3: 错误处理")
    code3 = """
x = 10 / 0  # 除零错误
"""
    result3 = executor.execute(code3)
    print(f"成功: {result3['success']}")
    print(f"错误: {result3['error']}")
    print()

    # 示例4：安全检查
    print("示例4: 安全检查")
    code4 = """
import os  # 禁止的导入
os.system('ls')
"""
    result4 = executor.execute(code4)
    print(f"成功: {result4['success']}")
    print(f"错误: {result4['error']}")


if __name__ == "__main__":
    example_usage()
