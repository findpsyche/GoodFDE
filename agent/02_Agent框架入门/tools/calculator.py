"""
计算器工具
提供数学计算功能
"""

import math
from typing import Union


class Calculator:
    """计算器工具类"""

    def __init__(self):
        """初始化计算器"""
        self.safe_dict = {
            # 基本函数
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,

            # 数学函数
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,

            # 常量
            'pi': math.pi,
            'e': math.e,
        }

    def calculate(self, expression: str) -> Union[float, int, str]:
        """
        计算数学表达式

        Args:
            expression: 数学表达式字符串

        Returns:
            计算结果或错误信息
        """
        try:
            # 清理表达式
            expression = expression.strip()

            # 安全计算
            result = eval(expression, {"__builtins__": {}}, self.safe_dict)

            # 格式化结果
            if isinstance(result, float):
                # 如果是整数，返回整数
                if result.is_integer():
                    return int(result)
                # 否则保留4位小数
                return round(result, 4)

            return result

        except ZeroDivisionError:
            return "错误: 除数不能为零"
        except SyntaxError:
            return "错误: 表达式语法错误"
        except NameError as e:
            return f"错误: 未知的函数或变量 - {e}"
        except Exception as e:
            return f"错误: {str(e)}"

    def get_description(self) -> str:
        """获取工具描述"""
        return """计算器工具 - 支持以下功能:

基本运算:
  • 加减乘除: +, -, *, /
  • 幂运算: **
  • 括号: ()

数学函数:
  • sqrt(x) - 平方根
  • sin(x), cos(x), tan(x) - 三角函数
  • log(x), log10(x) - 对数
  • exp(x) - 指数
  • abs(x) - 绝对值
  • round(x) - 四舍五入
  • pow(x, y) - x的y次方

常量:
  • pi - 圆周率
  • e - 自然常数

示例:
  • 2 + 2
  • sqrt(16)
  • sin(pi/2)
  • pow(2, 10)
"""


# 便捷函数
def calculate(expression: str) -> Union[float, int, str]:
    """
    便捷的计算函数

    Args:
        expression: 数学表达式

    Returns:
        计算结果
    """
    calc = Calculator()
    return calc.calculate(expression)


# 测试代码
if __name__ == "__main__":
    calc = Calculator()

    print("🧮 计算器工具测试\n")
    print(calc.get_description())

    test_cases = [
        "2 + 2",
        "10 * 5 + 3",
        "sqrt(16)",
        "sin(pi/2)",
        "pow(2, 10)",
        "log(e)",
        "(100 + 50) / 3",
        "10 / 0",  # 错误测试
        "invalid",  # 错误测试
    ]

    print("\n测试用例:")
    print("=" * 50)

    for expr in test_cases:
        result = calc.calculate(expr)
        print(f"{expr:20} = {result}")
