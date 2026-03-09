"""
Multi-Agent基础示例

Multi-Agent系统是多个Agent协同工作完成复杂任务的系统。
每个Agent有自己的专业领域和职责，通过协作达成共同目标。

核心特点：
1. 专业分工：每个Agent专注于特定任务
2. 协同工作：Agent之间相互配合
3. 结果汇总：整合各Agent的输出

设计原则：
- 验证驱动：验证每个Agent的输出
- 闭环思维：Agent间的反馈循环
- 降低认知负荷：清晰的协作协议
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
# 基础Agent类
# ============================================================================

class BaseAgent:
    """基础Agent类"""

    def __init__(self, name: str, role: str, llm: Optional[Any] = None, verbose: bool = False):
        """
        初始化Agent

        Args:
            name: Agent名称
            role: Agent角色描述
            llm: 语言模型
            verbose: 是否显示详细日志
        """
        self.name = name
        self.role = role
        self.llm = llm or self._get_default_llm()
        self.verbose = verbose

        logger.info(f"✅ {self.name} 初始化完成")

    def _get_default_llm(self):
        """获取默认LLM"""
        if os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.3)
        elif os.getenv("ANTHROPIC_API_KEY"):
            return ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.3)
        else:
            raise ValueError("请设置API密钥")

    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        处理输入数据

        Args:
            input_data: 输入数据

        Returns:
            处理结果
        """
        raise NotImplementedError("子类必须实现process方法")


# ============================================================================
# 专业Agent实现
# ============================================================================

class ResearchAgent(BaseAgent):
    """研究员Agent - 负责收集和分析信息"""

    def __init__(self, llm: Optional[Any] = None, verbose: bool = False):
        super().__init__(
            name="研究员",
            role="收集和分析信息",
            llm=llm,
            verbose=verbose
        )

    def process(self, topic: str) -> Dict[str, Any]:
        """
        研究主题

        Args:
            topic: 研究主题

        Returns:
            研究结果
        """
        logger.info(f"🔍 {self.name} 开始研究: {topic}")

        prompt = PromptTemplate.from_template("""
你是一个专业的研究员，需要收集和分析信息。

研究主题：{topic}

请提供：
1. 主题概述
2. 关键要点（3-5个）
3. 重要发现
4. 相关资源

格式：
## 概述
[概述内容]

## 关键要点
- 要点1
- 要点2
- 要点3

## 重要发现
[发现内容]

## 相关资源
- 资源1
- 资源2

研究报告：
""")

        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        research_result = chain.run(topic=topic)

        logger.info(f"✅ {self.name} 完成研究")

        return {
            'agent': self.name,
            'success': True,
            'topic': topic,
            'research': research_result,
            'length': len(research_result)
        }


class WriterAgent(BaseAgent):
    """写作Agent - 负责整理和撰写内容"""

    def __init__(self, llm: Optional[Any] = None, verbose: bool = False):
        super().__init__(
            name="写作者",
            role="整理和撰写内容",
            llm=llm,
            verbose=verbose
        )

    def process(self, research_data: str) -> Dict[str, Any]:
        """
        撰写内容

        Args:
            research_data: 研究数据

        Returns:
            撰写结果
        """
        logger.info(f"✍️  {self.name} 开始撰写")

        prompt = PromptTemplate.from_template("""
你是一个专业的写作者，需要将研究内容整理成易读的文章。

研究内容：
{research_data}

请撰写一篇结构清晰、内容完整的文章，包括：
1. 引言
2. 主体内容（分段）
3. 总结

要求：
- 使用Markdown格式
- 语言流畅、易懂
- 逻辑清晰
- 适当使用列表和强调

文章：
""")

        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        article = chain.run(research_data=research_data)

        logger.info(f"✅ {self.name} 完成撰写")

        return {
            'agent': self.name,
            'success': True,
            'article': article,
            'length': len(article)
        }


class ReviewerAgent(BaseAgent):
    """审核Agent - 负责审核和评估内容"""

    def __init__(self, llm: Optional[Any] = None, verbose: bool = False):
        super().__init__(
            name="审核员",
            role="审核和评估内容",
            llm=llm,
            verbose=verbose
        )

    def process(self, article: str) -> Dict[str, Any]:
        """
        审核文章

        Args:
            article: 文章内容

        Returns:
            审核结果
        """
        logger.info(f"📋 {self.name} 开始审核")

        prompt = PromptTemplate.from_template("""
你是一个严格的审核员，需要评估文章质量。

文章内容：
{article}

请评估以下方面（每项0-10分）：
1. 完整性
2. 准确性
3. 清晰度
4. 结构性
5. 整体质量

格式：
## 评分
- 完整性: X/10
- 准确性: X/10
- 清晰度: X/10
- 结构性: X/10
- 整体质量: X/10
- 总分: X/10

## 评价
[简短评价]

## 建议
- 建议1
- 建议2

审核结果：
""")

        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        review_result = chain.run(article=article)

        # 解析评分
        score = self._parse_score(review_result)
        approved = score >= 7.0

        logger.info(f"✅ {self.name} 完成审核: {score}/10 ({'通过' if approved else '不通过'})")

        return {
            'agent': self.name,
            'success': True,
            'review': review_result,
            'score': score,
            'approved': approved
        }

    def _parse_score(self, review_text: str) -> float:
        """解析总分"""
        lines = review_text.split('\n')
        for line in lines:
            if '总分:' in line or '总分：' in line:
                try:
                    score_str = line.split(':')[1].split('/')[0].strip()
                    return float(score_str)
                except:
                    pass
        return 5.0


# ============================================================================
# Multi-Agent系统
# ============================================================================

class MultiAgentSystem:
    """Multi-Agent系统 - 协调多个Agent协同工作"""

    def __init__(self, verbose: bool = False):
        """
        初始化Multi-Agent系统

        Args:
            verbose: 是否显示详细日志
        """
        self.verbose = verbose
        self.agents = {}

        logger.info("✅ Multi-Agent系统初始化完成")

    def register_agent(self, agent: BaseAgent):
        """
        注册Agent

        Args:
            agent: Agent实例
        """
        self.agents[agent.name] = agent
        logger.info(f"✅ 注册Agent: {agent.name} ({agent.role})")

    def run_sequential(self, task: str) -> Dict[str, Any]:
        """
        顺序执行模式

        Args:
            task: 任务描述

        Returns:
            执行结果
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"开始执行任务（顺序模式）: {task}")
        logger.info(f"{'='*70}\n")

        results = {}

        try:
            # 步骤1：研究员收集信息
            if '研究员' in self.agents:
                research_result = self.agents['研究员'].process(task)
                results['research'] = research_result

            # 步骤2：写作者撰写文章
            if '写作者' in self.agents and 'research' in results:
                writer_result = self.agents['写作者'].process(
                    results['research']['research']
                )
                results['writing'] = writer_result

            # 步骤3：审核员审核
            if '审核员' in self.agents and 'writing' in results:
                review_result = self.agents['审核员'].process(
                    results['writing']['article']
                )
                results['review'] = review_result

            logger.info(f"\n{'='*70}")
            logger.info("✅ 任务执行完成")
            logger.info(f"{'='*70}\n")

            return {
                'success': True,
                'task': task,
                'results': results,
                'final_output': results.get('writing', {}).get('article', ''),
                'approved': results.get('review', {}).get('approved', False)
            }

        except Exception as e:
            logger.error(f"❌ 任务执行失败: {e}")
            return {
                'success': False,
                'task': task,
                'error': str(e),
                'results': results
            }

    def run_with_feedback(self, task: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        带反馈的执行模式

        Args:
            task: 任务描述
            max_iterations: 最大迭代次数

        Returns:
            执行结果
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"开始执行任务（反馈模式）: {task}")
        logger.info(f"最大迭代次数: {max_iterations}")
        logger.info(f"{'='*70}\n")

        for iteration in range(max_iterations):
            logger.info(f"\n{'─'*70}")
            logger.info(f"迭代 {iteration + 1}/{max_iterations}")
            logger.info(f"{'─'*70}\n")

            # 执行完整流程
            result = self.run_sequential(task)

            if not result['success']:
                logger.warning(f"迭代 {iteration + 1} 失败")
                continue

            # 检查是否通过审核
            if result.get('approved', False):
                logger.info(f"✅ 审核通过！")
                result['iterations'] = iteration + 1
                return result

            # 未通过，提取反馈
            if iteration < max_iterations - 1:
                review_text = result['results']['review']['review']
                logger.info(f"⚠️  审核未通过，准备改进...")

                # 提取改进建议
                suggestions = self._extract_suggestions(review_text)
                logger.info(f"改进建议: {', '.join(suggestions[:2])}")

                # 更新任务（添加反馈）
                task = f"{task}\n\n请注意：{'; '.join(suggestions[:2])}"

        logger.warning(f"❌ 达到最大迭代次数，未能通过审核")
        result['iterations'] = max_iterations
        return result

    def _extract_suggestions(self, review_text: str) -> List[str]:
        """提取改进建议"""
        suggestions = []
        lines = review_text.split('\n')
        in_suggestions = False

        for line in lines:
            if '建议' in line and '##' in line:
                in_suggestions = True
                continue
            if in_suggestions:
                if line.startswith('##'):
                    break
                if line.strip().startswith('-') or line.strip().startswith('*'):
                    suggestions.append(line.strip().lstrip('-* '))

        return suggestions if suggestions else ["提高内容质量"]


# ============================================================================
# 示例函数
# ============================================================================

def example_1_sequential():
    """示例1：顺序执行"""
    print("\n" + "="*70)
    print("示例1: Multi-Agent - 顺序执行")
    print("="*70 + "\n")

    # 创建系统
    system = MultiAgentSystem(verbose=False)

    # 注册Agent
    system.register_agent(ResearchAgent())
    system.register_agent(WriterAgent())
    system.register_agent(ReviewerAgent())

    # 执行任务
    task = "Python装饰器的使用"

    result = system.run_sequential(task)

    if result['success']:
        print(f"\n✅ 任务完成")
        print(f"审核结果: {'通过' if result['approved'] else '不通过'}")
        print(f"\n最终文章:\n{result['final_output'][:300]}...")


def example_2_with_feedback():
    """示例2：带反馈的执行"""
    print("\n" + "="*70)
    print("示例2: Multi-Agent - 带反馈循环")
    print("="*70 + "\n")

    # 创建系统
    system = MultiAgentSystem(verbose=False)

    # 注册Agent
    system.register_agent(ResearchAgent())
    system.register_agent(WriterAgent())
    system.register_agent(ReviewerAgent())

    # 执行任务
    task = "解释什么是AI Agent"

    result = system.run_with_feedback(task, max_iterations=2)

    if result['success']:
        print(f"\n✅ 任务完成")
        print(f"迭代次数: {result.get('iterations', 0)}")
        print(f"审核结果: {'通过' if result['approved'] else '不通过'}")


def example_3_agent_roles():
    """示例3：展示Agent角色"""
    print("\n" + "="*70)
    print("示例3: Multi-Agent - Agent角色展示")
    print("="*70 + "\n")

    # 创建Agent
    agents = [
        ResearchAgent(),
        WriterAgent(),
        ReviewerAgent()
    ]

    print("系统中的Agent:")
    for agent in agents:
        print(f"  - {agent.name}: {agent.role}")

    print("\n工作流程:")
    print("  1. 研究员 → 收集信息")
    print("  2. 写作者 → 撰写文章")
    print("  3. 审核员 → 审核质量")
    print("  4. 如果不通过 → 返回步骤2")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("\n" + "="*70)
    print("Multi-Agent基础示例 - 多Agent协同工作")
    print("="*70)
    print("\n核心特点：")
    print("  ✅ 专业分工：每个Agent专注于特定任务")
    print("  ✅ 协同工作：Agent之间相互配合")
    print("  ✅ 结果汇总：整合各Agent的输出")
    print("\n" + "="*70 + "\n")

    # 检查API密钥
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("❌ 错误：请设置API密钥")
        return

    try:
        # 运行示例
        example_3_agent_roles()

        input("\n按Enter运行示例1...")
        example_1_sequential()

        input("\n按Enter运行示例2...")
        example_2_with_feedback()

        print("\n" + "="*70)
        print("✅ 所有示例运行完成！")
        print("="*70)
        print("\n💡 学习要点：")
        print("  1. Multi-Agent系统通过专业分工提高效率")
        print("  2. Agent之间可以传递信息和反馈")
        print("  3. 可以实现复杂的协作流程")
        print("  4. 适合需要多个专业角色的任务")
        print("\n📝 下一步：")
        print("  - 学习更复杂的Multi-Agent协作模式")
        print("  - 实现Agent间的并行执行")
        print("  - 添加更多专业Agent")
        print("\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)
        print(f"\n❌ 程序执行失败: {e}")


if __name__ == "__main__":
    main()
