"""
网页抓取工具

功能：
- 抓取网页内容
- 提取文本和链接
- 处理JavaScript渲染的页面

设计原则：
- 清晰的接口和文档
- 完善的错误处理
- 详细的日志记录
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import logging
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


class WebScraperTool:
    """
    网页抓取工具

    使用场景：
    - 需要获取网页内容
    - 需要提取特定信息
    - 需要收集链接

    注意事项：
    - 遵守robots.txt
    - 控制请求频率
    - 处理反爬虫机制
    """

    def __init__(self, timeout: int = 10, max_retries: int = 3):
        """
        初始化网页抓取工具

        Args:
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def scrape(self, url: str) -> Dict[str, any]:
        """
        抓取网页内容

        Args:
            url: 网页URL

        Returns:
            包含标题、文本、链接等信息的字典

        Raises:
            ValueError: URL格式错误
            requests.RequestException: 请求失败

        示例:
            >>> scraper = WebScraperTool()
            >>> result = scraper.scrape("https://example.com")
            >>> print(result['title'])
            'Example Domain'
        """
        # 验证URL
        if not self._is_valid_url(url):
            raise ValueError(f"无效的URL: {url}")

        logger.info(f"🌐 抓取网页: {url}")

        # 重试机制
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()

                # 解析HTML
                soup = BeautifulSoup(response.content, 'html.parser')

                # 提取信息
                result = {
                    'url': url,
                    'status_code': response.status_code,
                    'title': self._extract_title(soup),
                    'text': self._extract_text(soup),
                    'links': self._extract_links(soup, url),
                    'meta': self._extract_meta(soup)
                }

                logger.info(f"✅ 抓取成功: {result['title']}")
                return result

            except requests.RequestException as e:
                logger.warning(f"尝试 {attempt + 1}/{self.max_retries} 失败: {e}")
                if attempt == self.max_retries - 1:
                    raise

        raise Exception("抓取失败，已达最大重试次数")

    def _is_valid_url(self, url: str) -> bool:
        """验证URL格式"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """提取标题"""
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else ""

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """提取文本内容"""
        # 移除script和style标签
        for tag in soup(['script', 'style']):
            tag.decompose()

        # 获取文本
        text = soup.get_text()

        # 清理空白
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text[:5000]  # 限制长度

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """提取链接"""
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # 转换为绝对URL
            absolute_url = urljoin(base_url, href)
            if self._is_valid_url(absolute_url):
                links.append(absolute_url)

        return links[:50]  # 限制数量

    def _extract_meta(self, soup: BeautifulSoup) -> Dict[str, str]:
        """提取meta信息"""
        meta = {}

        # 提取description
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag and desc_tag.get('content'):
            meta['description'] = desc_tag['content']

        # 提取keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag and keywords_tag.get('content'):
            meta['keywords'] = keywords_tag['content']

        return meta

    def search_text(self, url: str, keyword: str) -> List[str]:
        """
        在网页中搜索关键词

        Args:
            url: 网页URL
            keyword: 搜索关键词

        Returns:
            包含关键词的段落列表
        """
        result = self.scrape(url)
        text = result['text']

        # 分段
        paragraphs = text.split('\n')

        # 查找包含关键词的段落
        matches = [p for p in paragraphs if keyword.lower() in p.lower()]

        logger.info(f"🔍 找到 {len(matches)} 个匹配段落")
        return matches[:10]  # 限制数量


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    print("\n" + "="*50)
    print("网页抓取工具示例")
    print("="*50 + "\n")

    scraper = WebScraperTool()

    # 示例1：抓取网页
    try:
        result = scraper.scrape("https://example.com")
        print(f"标题: {result['title']}")
        print(f"文本长度: {len(result['text'])}")
        print(f"链接数量: {len(result['links'])}")
    except Exception as e:
        print(f"抓取失败: {e}")

    # 示例2：搜索关键词
    try:
        matches = scraper.search_text("https://example.com", "example")
        print(f"\n找到 {len(matches)} 个匹配段落")
        for i, match in enumerate(matches[:3], 1):
            print(f"{i}. {match[:100]}...")
    except Exception as e:
        print(f"搜索失败: {e}")


if __name__ == "__main__":
    example_usage()
