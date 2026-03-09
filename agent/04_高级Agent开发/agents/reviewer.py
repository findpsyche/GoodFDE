"""
审核员Agent

职责：
- 评估内容质量
- 提供改进建议
- 决定是否通过

设计原则：
- 验证驱动：明确的评估标准
- 闭环思维：提供可操作的反馈
- 降低认知负荷：清晰的评分体系
"""

from typing import Dict, Any, List, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class ReviewerAgent:
    """
    审核员Agent

    使用场景：
    - 需要评估内容质量
    - 需要提供改进建议
    - 需要决策是否通过

    设计层次：
    - ✅ 及格线：能评分、能提建议
    - 🌟 状元：标准明确、反馈可操作
    """

    def __init__(self, llm: Optional[Any] = None, verbose: bool = True):
        """
        初始化审核员Agent

        Args:
            llm: 语言模型
            verbose: 是否显示详细日志
        """
        self.llm = llm or ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
        self.verbose = verbose

        # 评估标准
        self.criteria = {
            'completeness': '内容完整性（是否涵盖主题的关键方面）',
            'accuracy': '准确性（信息是否准确可靠）',
            'clarity': '清晰度（表达是否清晰易懂）',
            'structure': '结构性（组织是否合理）',
            'quality': '整体质量（语言、格式等）'
        }

        logger.info("✅ 审核员Agent初始化完成")

    def review(self, content: str, min_score: float = 7.0) -> Dict[str, Any]:
        """
        审核内容

        Args:
            content: 待审核内容
            min_score: 最低通过分数

        Returns:
            审核结果

        示例:
            >>> reviewer = ReviewerAgent()
            >>> result = reviewer.review("这是一篇文章...")
            >>> print(result['approved'])
            True
        """
        logger.info(f"📋 开始审核 (最低分数: {min_score})")

        try:
            # 创建审核prompt
            prompt = self._create_review_prompt()

            # 执行审核
            review_text = self.llm.invoke(
                prompt.format(
                    content=content,
                    criteria=self._format_criteria()
                )
            ).content

            # 解析审核结果
            scores = self._parse_scores(review_text)
            feedback = self._extract_feedback(review_text)
            overall_score = sum(scores.values()) / len(scores)

            # 判断是否通过
            approved = overall_score >= min_score

            logger.info(f"✅ 审核完成: {overall_score:.1f}/10 ({'通过' if approved else '不通过'})")

            return {
                'success': True,
                'approved': approved,
                'overall_score': overall_score,
                'scores': scores,
                'feedback': feedback,
                'review_text': review_text
            }

        except Exception as e:
            logger.error(f"❌ 审核失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _create_review_prompt(self) -> PromptTemplate:
        """创建审核prompt"""
        return PromptTemplate.from_template("""
你是一个专业的审核员，需要评估内容质量。

评估标准：
{criteria}

请按照以下格式审核内容：

## 评分（每项0-10分）
- 完整性: [分数]/10
- 准确性: [分数]/10
- 清晰度: [分数]/10
- 结构性: [分数]/10
- 整体质量: [分数]/10

## 优点
- 优点1
- 优点2

## 需要改进的地方
- 问题1: 具体建议
- 问题2: 具体建议

## 总体评价
[总体评价]

待审核内容：
{content}

审核结果：
""")

    def _format_criteria(self) -> str:
        """格式化评估标准"""
        return '\n'.join([f"- {name}: {desc}" for name, desc in self.criteria.items()])

    def _parse_scores(self, review_text: str) -> Dict[str, float]:
        """
        解析评分

        Args:
            review_text: 审核文本

        Returns:
            评分字典
        """
        scores = {}
        lines = review_text.split('\n')

        for line in lines:
            line = line.strip()
            if ':' in line and '/10' in line:
                # 提取分数
                try:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        score_part = parts[1].split('/')[0].strip()
                        score = float(score_part)
                        # 提取标准名称
                        criterion = parts[0].strip('- ').lower()
                        scores[criterion] = score
                except (ValueError, IndexError):
                    continue

        # 如果没有解析到分数，使用默认值
        if not scores:
            scores = {k: 7.0 for k in self.criteria.keys()}

        return scores

    def _extract_feedback(self, review_text: str) -> List[str]:
        """
        提取反馈建议

        Args:
            review_text: 审核文本

        Returns:
            反馈列表
        """
        feedback = []
        lines = review_text.split('\n')

        in_improvement_section = False
        for line in lines:
            line = line.strip()

            if '需要改进' in line or '改进建议' in line:
                in_improvement_section = True
                continue

            if in_improvement_section:
                if line.startswith('-') or line.startswith('*'):
                    feedback.append(line.lstrip('-* '))
                elif line.startswith('##'):
                    break

        return feedback

    def quick_review(self, content: str) -> Dict[str, Any]:
        """
        快速审核（简化版）

        Args:
            content: 待审核内容

        Returns:
            审核结果
        """
        logger.info("⚡ 快速审核")

        # 简单的启发式评估
        score = 0.0

        # 长度检查
        if len(content) > 500:
            score += 2.0
        elif len(content) > 200:
            score += 1.0

        # 结构检查
        if '##' in content:
            score += 2.0
        if '-' in content or '*' in content:
            score += 2.0

        # 格式检查
        if '```' in content:
            score += 1.0
        if content.count('\n\n') > 2:
            score += 2.0

        # 内容检查
        keywords = ['因为', '所以', '例如', '首先', '其次', '最后', '总结']
        keyword_count = sum(1 for kw in keywords if kw in content)
        score += min(keyword_count * 0.5, 1.0)

        approved = score >= 7.0

        logger.info(f"✅ 快速审核完成: {score:.1f}/10")

        return {
            'success': True,
            'approved': approved,
            'overall_score': score,
            'method': 'quick'
        }


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    print("\n" + "="*50)
    print("审核员Agent示例")
    print("="*50 + "\n")

    reviewer = ReviewerAgent(verbose=False)

    # 示例内容
    content = """
## Python最佳实践

Python是一种强大的编程语言，以下是一些最佳实践：

### 1. 代码风格
- 遵循PEP 8规范
- 使用有意义的变量名
- 添加适当的注释

### 2. 错误处理
- 使用try-except处理异常
- 提供清晰的错误信息
- 避免捕获所有异常

### 3. 性能优化
- 使用列表推导式
- 避免不必要的循环
- 使用生成器处理大数据

## 总结
遵循这些最佳实践可以提高代码质量和可维护性。
"""

    # 示例1：完整审核
    print("示例1: 完整审核")
    result = reviewer.review(content, min_score=7.0)
    if result['success']:
        print(f"通过: {result['approved']}")
        print(f"总分: {result['overall_score']:.1f}/10")
        print(f"反馈数量: {len(result['feedback'])}")
    print()

    # 示例2：快速审核
    print("示例2: 快速审核")
    result = reviewer.quick_review(content)
    print(f"通过: {result['approved']}")
    print(f"总分: {result['overall_score']:.1f}/10")


if __name__ == "__main__":
    example_usage()
