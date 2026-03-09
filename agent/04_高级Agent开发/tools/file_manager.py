"""
文件管理工具

功能：
- 读取文件内容
- 写入文件
- 列出目录
- 搜索文件

设计原则：
- 安全的文件操作
- 清晰的权限控制
- 详细的操作日志
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import logging
import json

logger = logging.getLogger(__name__)


class FileManagerTool:
    """
    文件管理工具

    使用场景：
    - 需要读取文件内容
    - 需要保存数据到文件
    - 需要管理文件和目录

    注意事项：
    - 限制访问范围
    - 验证文件路径
    - 处理权限错误
    """

    def __init__(self, base_dir: Optional[str] = None, max_file_size: int = 10 * 1024 * 1024):
        """
        初始化文件管理工具

        Args:
            base_dir: 基础目录（限制访问范围）
            max_file_size: 最大文件大小（字节）
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.max_file_size = max_file_size

    def read_file(self, file_path: str, encoding: str = 'utf-8') -> Dict[str, any]:
        """
        读取文件内容

        Args:
            file_path: 文件路径
            encoding: 文件编码

        Returns:
            包含内容和元信息的字典

        Raises:
            ValueError: 文件路径不安全
            FileNotFoundError: 文件不存在
            PermissionError: 没有读取权限

        示例:
            >>> manager = FileManagerTool()
            >>> result = manager.read_file("example.txt")
            >>> print(result['content'])
        """
        # 验证路径
        full_path = self._validate_path(file_path)

        logger.info(f"📖 读取文件: {full_path}")

        # 检查文件大小
        file_size = full_path.stat().st_size
        if file_size > self.max_file_size:
            raise ValueError(f"文件太大: {file_size} 字节 (最大 {self.max_file_size})")

        # 读取文件
        try:
            with open(full_path, 'r', encoding=encoding) as f:
                content = f.read()

            logger.info(f"✅ 读取成功: {len(content)} 字符")

            return {
                'success': True,
                'content': content,
                'size': file_size,
                'path': str(full_path)
            }

        except Exception as e:
            logger.error(f"❌ 读取失败: {e}")
            raise

    def write_file(self, file_path: str, content: str, encoding: str = 'utf-8') -> Dict[str, any]:
        """
        写入文件

        Args:
            file_path: 文件路径
            content: 文件内容
            encoding: 文件编码

        Returns:
            操作结果

        示例:
            >>> manager = FileManagerTool()
            >>> result = manager.write_file("output.txt", "Hello, World!")
            >>> print(result['success'])
            True
        """
        # 验证路径
        full_path = self._validate_path(file_path)

        logger.info(f"✍️  写入文件: {full_path}")

        # 创建父目录
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入文件
        try:
            with open(full_path, 'w', encoding=encoding) as f:
                f.write(content)

            logger.info(f"✅ 写入成功: {len(content)} 字符")

            return {
                'success': True,
                'path': str(full_path),
                'size': len(content)
            }

        except Exception as e:
            logger.error(f"❌ 写入失败: {e}")
            raise

    def list_files(self, directory: str = ".", pattern: str = "*") -> List[Dict[str, any]]:
        """
        列出目录中的文件

        Args:
            directory: 目录路径
            pattern: 文件模式（如 "*.txt"）

        Returns:
            文件信息列表
        """
        # 验证路径
        dir_path = self._validate_path(directory)

        logger.info(f"📂 列出目录: {dir_path}")

        files = []
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size': stat.st_size,
                    'modified': stat.st_mtime
                })

        logger.info(f"✅ 找到 {len(files)} 个文件")
        return files

    def search_files(self, keyword: str, directory: str = ".", pattern: str = "*.txt") -> List[Dict[str, any]]:
        """
        搜索包含关键词的文件

        Args:
            keyword: 搜索关键词
            directory: 搜索目录
            pattern: 文件模式

        Returns:
            匹配的文件列表
        """
        logger.info(f"🔍 搜索关键词: {keyword}")

        matches = []
        files = self.list_files(directory, pattern)

        for file_info in files:
            try:
                result = self.read_file(file_info['path'])
                if keyword.lower() in result['content'].lower():
                    matches.append({
                        'file': file_info['name'],
                        'path': file_info['path']
                    })
            except Exception as e:
                logger.warning(f"跳过文件 {file_info['name']}: {e}")

        logger.info(f"✅ 找到 {len(matches)} 个匹配文件")
        return matches

    def _validate_path(self, file_path: str) -> Path:
        """
        验证文件路径

        Args:
            file_path: 文件路径

        Returns:
            验证后的Path对象

        Raises:
            ValueError: 路径不安全
        """
        # 转换为绝对路径
        full_path = (self.base_dir / file_path).resolve()

        # 检查是否在允许的目录内
        try:
            full_path.relative_to(self.base_dir)
        except ValueError:
            raise ValueError(f"路径不在允许的范围内: {file_path}")

        return full_path


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    print("\n" + "="*50)
    print("文件管理工具示例")
    print("="*50 + "\n")

    # 创建临时目录
    import tempfile
    temp_dir = tempfile.mkdtemp()
    manager = FileManagerTool(base_dir=temp_dir)

    # 示例1：写入文件
    print("示例1: 写入文件")
    result1 = manager.write_file("test.txt", "Hello, World!")
    print(f"成功: {result1['success']}")
    print(f"路径: {result1['path']}")
    print()

    # 示例2：读取文件
    print("示例2: 读取文件")
    result2 = manager.read_file("test.txt")
    print(f"内容: {result2['content']}")
    print()

    # 示例3：列出文件
    print("示例3: 列出文件")
    files = manager.list_files()
    for file in files:
        print(f"  - {file['name']} ({file['size']} 字节)")
    print()

    # 示例4：搜索文件
    print("示例4: 搜索文件")
    manager.write_file("test2.txt", "Python is awesome")
    matches = manager.search_files("Python")
    for match in matches:
        print(f"  - {match['file']}")

    # 清理
    import shutil
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    example_usage()
