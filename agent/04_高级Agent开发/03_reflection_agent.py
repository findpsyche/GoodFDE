"""
Reflection模式示例

Reflection是一种自我反思和改进的Agent设计模式。
Agent在执行任务后会评估自己的表现，从失败中学习，并不断改进。

核心特点：
1. 自我评估：评估输出质量
2. 从失败中学习：分析错误原因
3. 迭代改进：根据反馈改进策略

设计原则：
- 验证驱动：明确的评估标准
- 闭环思维：评估-改进-再评估
- 降低认知负荷：清晰的反思框架
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


# ============================================================================
# Reflection Agent
# ============================================================================

class ReflectionAgent:
    """
    Reflection Agent - 自我反思和改进

    工作流程：
    1. Execute：执行任务
    2. Reflect：反思和评估
    3. Learn：从反馈中学习
    4. Improve：改进策略
    5. Retry：重新执行（如果需要）
    """

    def __init__(self, llm: Optional[Any] = None, verbose: bool = True):
        """
        初始化Reflection Agent

        Args:
            llm: 语言模型
            verbose: 是否显示详细日志
        """
        self.llm = llm or self._get_default_llm()
        self.verbose = verbose
        self.memory = []  # 存储经验

        logger.info("✅ Reflection Agent初始化完成")

    def _get_default_llm(self):
        """获取默认LLM"""
        if os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.3)
        elif os.getenv("ANTHROPIC_API_KEY"):
            return ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.3)
        else:
            raise ValueError("请设置API密钥")

    def run_with_reflection(
        self,
        task: str,
        max_iterations: int = 3,
        quality_threshold: float = 8.0
    ) -> Dict[str, Any]:
        """
        带反思的任务执行

        Args:
            task: 任务描述
            max_iterations: 最大迭代次数
            quality_threshold: 质量阈值

        Returns:
            执行结果
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"开始执行任务（带反思）: {task}")
        logger.info(f"质量阈值: {quality_threshold}/10")
        logger.info(f"{'='*70}\n")

        history = []

        for iteration in range(max_iterations):
            logger.info(f"\n{'─'*70}")
            logger.info(f"迭代 {iteration + 1}/{max_iterations}")
            logger.info(f"{'─'*70}\n")

            # 步骤1：执行任务
            logger.info("📝 步骤1: 执行任务")
            output = self._execute_task(task, history)
            logger.info(f"输出长度: {len(output)} 字符\n")

            # 步骤2：反思和评估
            logger.info("🤔 步骤2: 反思和评估")
            reflection = self._reflect(task, output, history)
            logger.info(f"质量评分: {reflection['score']}/10")
            logger.info(f"评价: {reflection['assessment']}\n")

            # 记录历史
            history.append({
                'iteration': iteration + 1,
                'output': output,
                'reflection': reflection
            })

            # 检查是否达标
            if reflection['score'] >= quality_threshold:
                logger.info(f"✅ 质量达标！")

                # 步骤3：学习经验
                self._learn_from_success(task, output, reflection)

                return {
                    'success': True,
                    'task': task,
                    'output': output,
                    'iterations': iteration + 1,
                    'quality_score': reflection['score'],
                    'history': history
                }

            # 步骤3：分析问题
            logger.info("🔍 步骤3: 分析问题")
            issues = self._analyze_issues(reflection)
            logger.info(f"发现 {len(issues)} 个问题:")
            for i, issue in enumerate(issues, 1):
                logger.info(f"  {i}. {issue}")
            logger.info()

            # 步骤4：生成改进建议
            logger.info("💡 步骤4: 生成改进建议")
            improvements = self._generate_improvements(issues)
            logger.info(f"改进建议:")
            for i, improvement in enumerate(improvements, 1):
                logger.info(f"  {i}. {improvement}")
            logger.info()

            # 步骤5：学习经验
            self._learn_from_failure(task, output, reflection, issues)

            # 如果不是最后一次迭代，准备下一次
            if iteration < max_iterations - 1:
                logger.info("🔄 准备下一次迭代...\n")
                # 更新任务描述，加入改进建议
                task = self._update_task_with_feedback(task, improvements)

        # 达到最大迭代次数
        logger.warning(f"\n⚠️  达到最大迭代次数，未能达到质量要求")

        return {
            'success': False,
            'task': task,
            'output': output,
            'iterations': max_iterations,
            'quality_score': reflection['score'],
            'history': history,
            'message': '未能达到质量要求'
        }

    def _execute_task(self, task: str, history: List[Dict]) -> str:
        """执行任务"""
        # 创建执行prompt
        prompt = PromptTemplate.from_template("""
你是一个专业的内容创作者，需要完成以下任务。

任务：{task}

{history_context}

请创作高质量的内容，确保：
1. 内容完整、准确
2. 结构清晰、逻辑性强
3. 语言流畅、易于理解
4. 使用Markdown格式

内容：
""")

        # 构建历史上下文
        history_context = ""
        if history:
            history_context = "之前的尝试和反馈：\n"
            for h in history[-2:]:  # 只使用最近2次
                history_context += f"\n尝试 {h['iteration']}:\n"
                history_context += f"- 评分: {h['reflection']['score']}/10\n"
                history_context += f"- 问题: {h['reflection']['issues']}\n"

        # 执行
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        output = chain.run(task=task, history_context=history_context)

        return output

    def _reflect(self, task: str, output: str, history: List[Dict]) -> Dict[str, Any]:
        """反思和评估"""
        prompt = PromptTemplate.from_template("""
你是一个严格的评审员，需要评估内容质量。

任务：{task}

输出内容：
{output}

请从以下维度评估（每项0-10分）：
1. 完整性：是否完整回答了任务要求
2. 准确性：信息是否准确可靠
3. 清晰度：表达是否清晰易懂
4. 结构性：组织是否合理
5. 质量：整体质量如何

请按以下格式输出：

## 评分
- 完整性: X/10
- 准确性: X/10
- 清晰度: X/10
- 结构性: X/10
- 质量: X/10
- 总分: X/10

## 评价
[简短评价]

## 优点
- 优点1
- 优点2

## 问题
- 问题1
- 问题2

## 改进建议
- 建议1
- 建议2

评估结果：
""")

        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        reflection_text = chain.run(task=task, output=output)

        # 解析反思结果
        score = self._parse_score(reflection_text)
        assessment = self._extract_section(reflection_text, "评价")
        strengths = self._extract_list(reflection_text, "优点")
        issues = self._extract_list(reflection_text, "问题")
        suggestions = self._extract_list(reflection_text, "改进建议")

        return {
            'score': score,
            'assessment': assessment,
            'strengths': strengths,
            'issues': issues,
            'suggestions': suggestions,
            'full_text': reflection_text
        }

    def _parse_score(self, text: str) -> float:
        """解析总分"""
        lines = text.split('\n')
        for line in lines:
            if '总分:' in line or '总分：' in line:
                try:
                    score_str = line.split(':')[1].split('/')[0].strip()
                    return float(score_str)
                except:
                    pass
        return 5.0  # 默认分数

    def _extract_section(self, text: str, section_name: str) -> str:
        """提取章节内容"""
        lines = text.split('\n')
        in_section = False
        content = []

        for line in lines:
            if section_name in line and '##' in line:
                in_section = True
                continue
            if in_section:
                if line.startswith('##'):
                    break
                if line.strip():
                    content.append(line.strip())

        return ' '.join(content) if content else ""

    def _extract_list(self, text: str, section_name: str) -> List[str]:
        """提取列表项"""
        lines = text.split('\n')
        in_section = False
        items = []

        for line in lines:
            if section_name in line and '##' in line:
                in_section = True
                continue
            if in_section:
                if line.startswith('##'):
                    break
                if line.strip().startswith('-') or line.strip().startswith('*'):
                    items.append(line.strip().lstrip('-* '))

        return items

    def _analyze_issues(self, reflection: Dict[str, Any]) -> List[str]:
        """分析问题"""
        return reflection.get('issues', [])

    def _generate_improvements(self, issues: List[str]) -> List[str]:
        """生成改进建议"""
        improvements = []

        for issue in issues:
            if '完整' in issue or '不足' in issue:
                improvements.append("增加更多细节和深度")
            elif '准确' in issue or '错误' in issue:
                improvements.append("验证信息的准确性")
            elif '清晰' in issue or '模糊' in issue:
                improvements.append("使用更清晰的表达")
            elif '结构' in issue or '组织' in issue:
                improvements.append("改进内容结构")
            else:
                improvements.append(f"解决：{issue}")

        return improvements if improvements else ["提高整体质量"]

    def _learn_from_success(self, task: str, output: str, reflection: Dict):
        """从成功中学习"""
        self.memory.append({
            'type': 'success',
            'task': task,
            'output_length': len(output),
            'score': reflection['score'],
            'strengths': reflection.get('strengths', [])
        })

        logger.info("📚 记录成功经验")

    def _learn_from_failure(self, task: str, output: str, reflection: Dict, issues: List[str]):
        """从失败中学习"""
        self.memory.append({
            'type': 'failure',
            'task': task,
            'output_length': len(output),
            'score': reflection['score'],
            'issues': issues
        })

        logger.info("📚 记录失败教训")

    def _update_task_with_feedback(self, task: str, improvements: List[str]) -> str:
        """根据反馈更新任务"""
        feedback = '\n'.join([f"- {imp}" for imp in improvements])
        return f"{task}\n\n请特别注意：\n{feedback}"

    def get_learned_lessons(self) -> Dict[str, Any]:
        """获取学到的经验"""
        successes = [m for m in self.memory if m['type'] == 'success']
        failures = [m for m in self.memory if m['type'] == 'failure']

        return {
            'total_experiences': len(self.memory),
            'successes': len(successes),
            'failures': len(failures),
            'success_rate': len(successes) / len(self.memory) if self.memory else 0,
            'common_issues': self._get_common_issues(failures),
            'success_patterns': self._get_success_patterns(successes)
        }

    def _get_common_issues(self, failures: List[Dict]) -> List[str]:
        """获取常见问题"""
        all_issues = []
        for f in failures:
            all_issues.extend(f.get('issues', []))

        # 简单统计（实际应该用更复杂的方法）
        return list(set(all_issues))[:5]

    def _get_success_patterns(self, successes: List[Dict]) -> List[str]:
        """获取成功模式"""
        all_strengths = []
        for s in successes:
            all_strengths.extend(s.get('strengths', []))

        return list(set(all_strengths))[:5]


# ============================================================================
# 示例函数
# ============================================================================

def example_1_basic():
    """示例1：基础Reflection"""
    print("\n" + "="*70)
    print("示例1: Reflection模式 - 基础")
    print("="*70 + "\n")

    agent = ReflectionAgent(verbose=False)

    task = "写一篇关于Python最佳实践的简短文章（200字左右）"

    result = agent.run_with_reflection(
        task=task,
        max_iterations=3,
        quality_threshold=7.0
    )

    if result['success']:
        print(f"\n✅ 成功！经过 {result['iterations']} 次迭代")
        print(f"最终评分: {result['quality_score']}/10")
        print(f"\n最终输出:\n{result['output'][:300]}...")
    else:
        print(f"\n⚠️  未达标，最终评分: {result['quality_score']}/10")


def example_2_learning():
    """示例2：学习和改进"""
    print("\n" + "="*70)
    print("示例2: Reflection模式 - 学习和改进")
    print("="*70 + "\n")

    agent = ReflectionAgent(verbose=False)

    # 执行多个任务
    tasks = [
        "解释什么是AI Agent",
        "介绍Python的主要特点",
        "说明如何学习编程"
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\n任务 {i}: {task}")
        result = agent.run_with_reflection(
            task=task,
            max_iterations=2,
            quality_threshold=7.0
        )
        print(f"结果: {'✅ 成功' if result['success'] else '⚠️  未达标'}")

    # 查看学到的经验
    print("\n" + "-"*70)
    print("学习总结:")
    lessons = agent.get_learned_lessons()
    print(f"总经验数: {lessons['total_experiences']}")
    print(f"成功次数: {lessons['successes']}")
    print(f"失败次数: {lessons['failures']}")
    print(f"成功率: {lessons['success_rate']*100:.1f}%")


def example_3_iteration_history():
    """示例3：查看迭代历史"""
    print("\n" + "="*70)
    print("示例3: Reflection模式 - 迭代历史")
    print("="*70 + "\n")

    agent = ReflectionAgent(verbose=False)

    task = "写一个Python函数的文档字符串示例"

    result = agent.run_with_reflection(
        task=task,
        max_iterations=3,
        quality_threshold=8.0
    )

    # 显示迭代历史
    print("\n迭代历史:")
    for h in result['history']:
        print(f"\n迭代 {h['iteration']}:")
        print(f"  评分: {h['reflection']['score']}/10")
        print(f"  评价: {h['reflection']['assessment'][:100]}...")
        if h['reflection']['issues']:
            print(f"  问题: {', '.join(h['reflection']['issues'][:2])}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("\n" + "="*70)
    print("Reflection模式示例 - 自我反思和改进")
    print("="*70)
    print("\n核心特点：")
    print("  ✅ 自我评估：评估输出质量")
    print("  ✅ 从失败中学习：分析错误原因")
    print("  ✅ 迭代改进：根据反馈改进策略")
    print("\n" + "="*70 + "\n")

    # 检查API密钥
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("❌ 错误：请设置API密钥")
        return

    try:
        # 运行示例
        example_1_basic()

        input("\n按Enter继续下一个示例...")
        example_2_learning()

        input("\n按Enter继续下一个示例...")
        example_3_iteration_history()

        print("\n" + "="*70)
        print("✅ 所有示例运行完成！")
        print("="*70)
        print("\n💡 学习要点：")
        print("  1. Reflection模式通过自我评估改进输出")
        print("  2. 每次迭代都会反思和学习")
        print("  3. 可以积累经验，避免重复错误")
        print("  4. 适合需要高质量输出的任务")
        print("\n📝 与其他模式对比：")
        print("  - ReAct: 边思考边行动")
        print("  - Plan-and-Execute: 先规划后执行")
        print("  - Reflection: 执行-反思-改进")
        print("\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)
        print(f"\n❌ 程序执行失败: {e}")


if __name__ == "__main__":
    main()
