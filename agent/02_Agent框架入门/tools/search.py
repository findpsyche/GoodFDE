"""
搜索工具
提供网络搜索功能（DuckDuckGo和Wikipedia）
"""

import os
from typing import List, Dict, Optional


class SearchTool:
    """搜索工具类"""

    def __init__(self):
        """初始化搜索工具"""
        self.max_results = 5

    def search_duckduckgo(self, query: str, max_results: int = 3) -> str:
        """
        使用DuckDuckGo搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数

        Returns:
            格式化的搜索结果
        """
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for i, result in enumerate(ddgs.text(query, max_results=max_results)):
                    results.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('body', ''),
                        'url': result.get('href', '')
                    })

            if not results:
                return f"未找到关于 '{query}' 的搜索结果"

            # 格式化输出
            output = f"🔍 搜索结果: {query}\n"
            output += "=" * 60 + "\n\n"

            for i, result in enumerate(results, 1):
                output += f"{i}. {result['title']}\n"
                output += f"   {result['snippet'][:150]}...\n"
                output += f"   🔗 {result['url']}\n\n"

            return output

        except ImportError:
            return "❌ 需要安装duckduckgo-search: pip install duckduckgo-search"
        except Exception as e:
            return f"❌ 搜索失败: {str(e)}"

    def search_wikipedia(self, query: str, sentences: int = 3) -> str:
        """
        搜索Wikipedia

        Args:
            query: 搜索查询
            sentences: 返回的句子数

        Returns:
            Wikipedia摘要
        """
        try:
            import wikipedia
            wikipedia.set_lang("zh")

            try:
                # 搜索并获取摘要
                summary = wikipedia.summary(query, sentences=sentences)

                output = f"📚 Wikipedia: {query}\n"
                output += "=" * 60 + "\n\n"
                output += summary + "\n\n"

                # 获取页面URL
                page = wikipedia.page(query)
                output += f"🔗 详细信息: {page.url}\n"

                return output

            except wikipedia.exceptions.DisambiguationError as e:
                # 如果有歧义，返回选项
                options = ', '.join(e.options[:5])
                return f"📚 找到多个相关主题:\n{options}\n\n请使用更具体的搜索词。"

            except wikipedia.exceptions.PageError:
                return f"❌ 未找到关于 '{query}' 的Wikipedia页面"

        except ImportError:
            return "❌ 需要安装wikipedia: pip install wikipedia"
        except Exception as e:
            return f"❌ Wikipedia搜索失败: {str(e)}"

    def search(self, query: str, source: str = "auto") -> str:
        """
        通用搜索接口

        Args:
            query: 搜索查询
            source: 搜索源 ("auto", "duckduckgo", "wikipedia")

        Returns:
            搜索结果
        """
        if source == "wikipedia":
            return self.search_wikipedia(query)
        elif source == "duckduckgo":
            return self.search_duckduckgo(query)
        else:
            # 自动选择：先尝试Wikipedia，如果失败则用DuckDuckGo
            wiki_result = self.search_wikipedia(query, sentences=2)
            if "❌" not in wiki_result:
                return wiki_result
            else:
                return self.search_duckduckgo(query)

    def get_description(self) -> str:
        """获取工具描述"""
        return """搜索工具 - 网络搜索和知识查询

功能:
  • DuckDuckGo网络搜索
  • Wikipedia知识查询
  • 自动选择最佳搜索源

使用方法:
  1. 通用搜索: search("查询内容")
  2. Wikipedia: search("查询内容", source="wikipedia")
  3. DuckDuckGo: search("查询内容", source="duckduckgo")

示例:
  • search("人工智能")
  • search("Python编程", source="wikipedia")
  • search("最新AI新闻", source="duckduckgo")

依赖安装:
  pip install duckduckgo-search wikipedia
"""


# 便捷函数
def search(query: str, source: str = "auto") -> str:
    """
    便捷的搜索函数

    Args:
        query: 搜索查询
        source: 搜索源

    Returns:
        搜索结果
    """
    tool = SearchTool()
    return tool.search(query, source)


def search_wikipedia(query: str) -> str:
    """Wikipedia搜索便捷函数"""
    tool = SearchTool()
    return tool.search_wikipedia(query)


def search_duckduckgo(query: str) -> str:
    """DuckDuckGo搜索便捷函数"""
    tool = SearchTool()
    return tool.search_duckduckgo(query)


# 测试代码
if __name__ == "__main__":
    tool = SearchTool()

    print("🔍 搜索工具测试\n")
    print(tool.get_description())

    print("\n" + "=" * 60)
    print("测试搜索:")
    print("=" * 60)

    # 测试Wikipedia
    print("\n1. Wikipedia搜索:")
    print("-" * 60)
    result = tool.search_wikipedia("LangChain")
    print(result)

    # 测试DuckDuckGo
    print("\n2. DuckDuckGo搜索:")
    print("-" * 60)
    result = tool.search_duckduckgo("AI Agent框架")
    print(result)

    # 测试自动选择
    print("\n3. 自动选择搜索源:")
    print("-" * 60)
    result = tool.search("人工智能")
    print(result)
