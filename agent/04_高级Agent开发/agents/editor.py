"""
编辑Agent

职责：
- 整理和优化内容
- 改进结构和可读性
- 确保内容质量

设计原则：
- 验证驱动：验证编辑质量
- 闭环思维：迭代改进内容
- 降低认知负荷：清晰的编辑规则
"""

from typing import Dict, Any, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class EditorAgent:
    """
    编辑Agent

    使用场景：
    - 需要整理研究内容
    - 需要改进文本质量
    - 需要统一格式和风格

    设计层次：
    - ✅ 及格线：能整理、能格式化、能优化
    - 🌟 状元：能评估质量、能迭代改进
    """

    def __init__(self, llm: Optional[Any] = None, verbose: bool = True):
        """
        初始化编辑Agent

        Args:
            llm: 语言模型
            verbose: 是否显示详细日志
        """
        self.llm = llm or ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.3)
        self.verbose = verbose

        logger.info("✅ 编辑Agent初始化完成")

    def edit(self, content: str, style: str = "professional") -> Dict[str, Any]:
        """
        编辑内容

        Args:
            content: 原始内容
            style: 编辑风格（professional, casual, academic）

        Returns:
            编辑结果

        示例:
            >>> editor = EditorAgent()
            >>> result = editor.edit("这是一些研究内容...")
            >>> print(result['edited_content'])
        """
        logger.info(f"✍️  开始编辑 ({style}风格)")

        try:
            # 创建编辑prompt
            prompt = self._create_edit_prompt(style)

            # 执行编辑
            edited_content = self.llm.invoke(
                prompt.format(content=content)
            ).content

            # 评估改进程度
            improvement_score = self._evaluate_improvement(content, edited_content)

            logger.info(f"✅ 编辑完成，改进评分: {improvement_score}/10")

            return {
                'success': True,
                'original_content': content,
                'edited_content': edited_content,
                'improvement_score': improvement_score,
                'style': style
            }

        except Exception as e:
            logger.error(f"❌ 编辑失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _create_edit_prompt(self, style: str) -> PromptTemplate:
        """创建编辑prompt"""
        style_guides = {
            'professional': """
你是一个专业的编辑，擅长整理和优化内容。

编辑要求：
1. 改进结构：使用清晰的标题和段落
2. 优化语言：使用专业、准确的表达
3. 增强可读性：使用列表、要点等格式
4. 保持完整性：不删除重要信息
5. 统一格式：使用Markdown格式

请编辑以下内容：

{content}

编辑后的内容：
""",
            'casual': """
你是一个友好的编辑，擅长让内容更易读。

编辑要求：
1. 使用简单、通俗的语言
2. 添加适当的例子和类比
3. 保持轻松、友好的语气
4. 使用短句和段落
5. 使用Markdown格式

请编辑以下内容：

{content}

编辑后的内容：
""",
            'academic': """
你是一个学术编辑，擅长学术写作。

编辑要求：
1. 使用正式、严谨的语言
2. 添加适当的引用和参考
3. 保持客观、中立的语气
4. 使用学术写作规范
5. 使用Markdown格式

请编辑以下内容：

{content}

编辑后的内容：
"""
        }

        template = style_guides.get(style, style_guides['professional'])
        return PromptTemplate.from_template(template)

    def _evaluate_improvement(self, original: str, edited: str) -> float:
        """
        评估改进程度

        Args:
            original: 原始内容
            edited: 编辑后内容

        Returns:
            改进评分（0-10）
        """
        score = 0.0

        # 检查长度变化（适度增加是好的）
        length_ratio = len(edited) / max(len(original), 1)
        if 1.0 <= length_ratio <= 1.5:
            score += 2.0
        elif 0.8 <= length_ratio < 1.0:
            score += 1.0

        # 检查结构改进
        if edited.count('##') > original.count('##'):
            score += 2.0
        if edited.count('-') > original.count('-'):
            score += 2.0

        # 检查格式
        if '```' in edited:  # 有代码块
            score += 1.0
        if '**' in edited or '*' in edited:  # 有强调
            score += 1.0

        # 检查段落
        edited_paragraphs = edited.count('\n\n')
        if edited_paragraphs > 0:
            score += 2.0

        return min(score, 10.0)

    def edit_with_feedback(self, content: str, feedback: str) -> Dict[str, Any]:
        """
        根据反馈编辑

        Args:
            content: 内容
            feedback: 反馈意见

        Returns:
            编辑结果
        """
        logger.info(f"✍️  根据反馈编辑")

        prompt = PromptTemplate.from_template("""
你是一个专业的编辑，需要根据反馈改进内容。

原始内容：
{content}

反馈意见：
{feedback}

请根据反馈改进内容，确保：
1. 解决反馈中提到的所有问题
2. 保持内容的完整性和准确性
3. 使用清晰的Markdown格式

改进后的内容：
""")

        try:
            edited_content = self.llm.invoke(
                prompt.format(content=content, feedback=feedback)
            ).content

            logger.info("✅ 根据反馈编辑完成")

            return {
                'success': True,
                'edited_content': edited_content,
                'feedback_applied': feedback
            }

        except Exception as e:
            logger.error(f"❌ 编辑失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    print("\n" + "="*50)
    print("编辑Agent示例")
    print("="*50 + "\n")

    editor = EditorAgent(verbose=False)

    # 示例内容
    content = """
Python是一种编程语言。它很好用。很多人用它。
它可以做很多事情。比如网站开发、数据分析、机器学习等。
Python有很多库。这些库很有用。
"""

    # 示例1：专业风格编辑
    print("示例1: 专业风格编辑")
    result = editor.edit(content, style="professional")
    if result['success']:
        print(f"改进评分: {result['improvement_score']}/10")
        print(f"\n编辑后内容:\n{result['edited_content'][:200]}...")
    print()

    # 示例2：根据反馈编辑
    print("示例2: 根据反馈编辑")
    feedback = "请添加具体的Python库示例，并使用列表格式"
    result = editor.edit_with_feedback(content, feedback)
    if result['success']:
        print(f"编辑后内容:\n{result['edited_content'][:200]}...")


if __name__ == "__main__":
    example_usage()
