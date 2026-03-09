#!/usr/bin/env python3
"""
测试脚本 - 验证环境配置和依赖安装
运行此脚本检查是否正确配置了开发环境
"""

import sys
import os
from typing import List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """检查Python版本"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        return True, f"✅ Python版本: {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro} (需要3.10+)"


def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """检查包是否安装"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        return True, f"✅ {package_name}"
    except ImportError:
        return False, f"❌ {package_name} 未安装"


def check_env_file() -> Tuple[bool, str]:
    """检查.env文件"""
    if os.path.exists('.env'):
        return True, "✅ .env文件存在"
    else:
        return False, "❌ .env文件不存在 (请复制.env.example为.env)"


def check_api_keys() -> List[Tuple[bool, str]]:
    """检查API密钥配置"""
    from dotenv import load_dotenv
    load_dotenv()

    results = []

    # 检查OpenAI
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key and openai_key != 'your_openai_api_key_here':
        results.append((True, "✅ OPENAI_API_KEY 已配置"))
    else:
        results.append((False, "⚠️  OPENAI_API_KEY 未配置"))

    # 检查Anthropic
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    if anthropic_key and anthropic_key != 'your_anthropic_api_key_here':
        results.append((True, "✅ ANTHROPIC_API_KEY 已配置"))
    else:
        results.append((False, "⚠️  ANTHROPIC_API_KEY 未配置"))

    return results


def test_langchain_basic() -> Tuple[bool, str]:
    """测试LangChain基础功能"""
    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import PromptTemplate

        # 创建简单的prompt
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="简单介绍{topic}"
        )

        result = prompt.format(topic="LangChain")

        if "LangChain" in result:
            return True, "✅ LangChain基础功能正常"
        else:
            return False, "❌ LangChain基础功能异常"
    except Exception as e:
        return False, f"❌ LangChain测试失败: {str(e)}"


def test_tools() -> List[Tuple[bool, str]]:
    """测试自定义工具"""
    results = []

    # 测试Calculator
    try:
        from tools.calculator import calculate
        result = calculate("2+2")
        if "4" in str(result):
            results.append((True, "✅ Calculator工具正常"))
        else:
            results.append((False, "❌ Calculator工具异常"))
    except Exception as e:
        results.append((False, f"❌ Calculator工具错误: {str(e)}"))

    # 测试Weather
    try:
        from tools.weather import get_weather
        result = get_weather("北京")
        if "北京" in result:
            results.append((True, "✅ Weather工具正常"))
        else:
            results.append((False, "❌ Weather工具异常"))
    except Exception as e:
        results.append((False, f"❌ Weather工具错误: {str(e)}"))

    # 测试Search
    try:
        from tools.search import SearchTool
        tool = SearchTool()
        # 只测试工具创建，不实际搜索
        results.append((True, "✅ Search工具正常"))
    except Exception as e:
        results.append((False, f"❌ Search工具错误: {str(e)}"))

    return results


def main():
    """主测试函数"""
    print("=" * 60)
    print("🔍 环境配置检查")
    print("=" * 60)

    all_passed = True

    # 1. Python版本
    print("\n1. Python环境")
    print("-" * 60)
    passed, msg = check_python_version()
    print(msg)
    all_passed = all_passed and passed

    # 2. 必需包
    print("\n2. 必需包检查")
    print("-" * 60)
    required_packages = [
        ("langchain", "langchain"),
        ("langchain-openai", "langchain_openai"),
        ("langchain-anthropic", "langchain_anthropic"),
        ("python-dotenv", "dotenv"),
        ("requests", "requests"),
    ]

    for package_name, import_name in required_packages:
        passed, msg = check_package(package_name, import_name)
        print(msg)
        all_passed = all_passed and passed

    # 3. 可选包
    print("\n3. 可选包检查")
    print("-" * 60)
    optional_packages = [
        ("duckduckgo-search", "duckduckgo_search"),
        ("wikipedia", "wikipedia"),
    ]

    for package_name, import_name in optional_packages:
        passed, msg = check_package(package_name, import_name)
        print(msg)
        # 可选包不影响总体结果

    # 4. 环境文件
    print("\n4. 环境配置")
    print("-" * 60)
    passed, msg = check_env_file()
    print(msg)

    if passed:
        # 检查API密钥
        api_results = check_api_keys()
        for passed, msg in api_results:
            print(msg)

    # 5. LangChain功能
    print("\n5. LangChain功能测试")
    print("-" * 60)
    passed, msg = test_langchain_basic()
    print(msg)
    all_passed = all_passed and passed

    # 6. 自定义工具
    print("\n6. 自定义工具测试")
    print("-" * 60)
    tool_results = test_tools()
    for passed, msg in tool_results:
        print(msg)
        all_passed = all_passed and passed

    # 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有必需检查通过！环境配置正确。")
        print("\n你可以开始运行示例代码了:")
        print("  python 01_langchain_basics.py")
        print("  python 02_langchain_agents.py")
        print("  python 03_memory_demo.py")
        print("  python 04_agent_with_tools.py")
    else:
        print("❌ 部分检查未通过，请根据上述提示修复问题。")
        print("\n常见问题解决:")
        print("  1. 安装依赖: pip install -r requirements.txt")
        print("  2. 配置环境: cp .env.example .env")
        print("  3. 填写API密钥: 编辑.env文件")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
