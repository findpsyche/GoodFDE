"""
研究员Agent

职责：
- 搜索和收集信息
- 分析和整理资料
- 提供研究报告

设计原则：
- 验证驱动：验证搜索结果质量
- 闭环思维：不断改进搜索策略
- 降低认知负荷：清晰的输出格式
"""

from typing import List, Dict, Any, Optional
import logging
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class ResearcherAgent:
    """
    研究员Agent

    使用场景：
    - 需要收集特定主题的信息
    - 需要分析多个来源
    - 需要生成研究报告

    设计层次：
    - ✅ 及格线：能搜索、能验证、能报告
    - 🌟 状元：能评估质量、能迭代改进
    """

    def __init__(self, tools: List[Tool], llm: Optional[Any] = None, verbose: bool = True):
        """
        初始化研究员Agent

        Args:
            tools: 可用工具列表
            llm: 语言模型
            verbose: 是否显示详细日志
        """
        self.tools = tools
        self.llm = llm or ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
        self.verbose = verbose

        # 创建agent
        self.agent = self._create_agent()

        logger.info("✅ 研究员Agent初始化完成")

    def _create_agent(self) -> AgentExecutor:
        """创建Agent执行器"""
        prompt = PromptTemplate.from_template("""
你是一个专业的研究员，擅长收集和分析信息。

你的职责：
1. 使用搜索工具收集相关信息
2. 分析信息的可靠性和相关性
3. 整理成结构化的研究报告

可用工具：
{tools}

工具名称：{tool_names}

请使用以下格式：

Question: 研究主题
Thought: 我需要搜索什么信息？
Action: 工具名称
Action Input: 工具输入
Observation: 工具输出
... (重复Thought/Action/Observation)
Thought: 我已经收集了足够的信息
Final Answer: 研究报告

研究报告格式：
## 主题
[主题名称]

## 关键发现
- 发现1
- 发现2
- 发现3

## 详细信息
[详细描述]

## 信息来源
- 来源1
- 来源2

开始！

Question: {input}
Thought: {agent_scratchpad}
""")

        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=10,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

    def research(self, topic: str) -> Dict[str, Any]:
        """
        执行研究任务

        Args:
            topic: 研究主题

        Returns:
            研究结果

        示例:
            >>> researcher = ResearcherAgent(tools=[search_tool])
            >>> result = researcher.research("Python最佳实践")
            >>> print(result['report'])
        """
        logger.info(f"🔍 开始研究: {topic}")

        try:
            # 执行研究
            result = self.agent.invoke({"input": topic})

            # 提取报告
            report = result["output"]

            # 验证报告质量
            quality_score = self._evaluate_quality(report)

            logger.info(f"✅ 研究完成，质量评分: {quality_score}/10")

            return {
                'success': True,
                'topic': topic,
                'report': report,
                'quality_score': quality_score,
                'steps': len(result.get('intermediate_steps', []))
            }

        except Exception as e:
            logger.error(f"❌ 研究失败: {e}")
            return {
                'success': False,
                'topic': topic,
                'error': str(e)
            }

    def _evaluate_quality(self, report: str) -> float:
        """
        评估报告质量

        Args:
            report: 研究报告

        Returns:
            质量评分（0-10）
        """
        score = 0.0

        # 检查长度
        if len(report) > 500:
            score += 3.0
        elif len(report) > 200:
            score += 2.0
        elif len(report) > 100:
            score += 1.0

        # 检查结构
        if "##" in report:  # 有标题
            score += 2.0
        if "-" in report or "*" in report:  # 有列表
            score += 2.0

        # 检查内容
        keywords = ["发现", "信息", "来源", "分析", "总结"]
        keyword_count = sum(1 for kw in keywords if kw in report)
        score += min(keyword_count, 3.0)

        return min(score, 10.0)

    def research_with_verification(self, topic: str, min_quality: float = 7.0, max_attempts: int = 3) -> Dict[str, Any]:
        """
        带验证的研究（闭环模式）

        Args:
            topic: 研究主题
            min_quality: 最低质量要求
            max_attempts: 最大尝试次数

        Returns:
            研究结果
        """
        logger.info(f"🔍 开始闭环研究: {topic} (最低质量: {min_quality})")

        for attempt in range(max_attempts):
            # 执行研究
            result = self.research(topic)

            if not result['success']:
                logger.warning(f"尝试 {attempt + 1} 失败")
                continue

            # 检查质量
            if result['quality_score'] >= min_quality:
                logger.info(f"✅ 质量达标: {result['quality_score']}/10")
                return result

            # 质量不够，改进主题
            logger.info(f"⚠️  质量不足: {result['quality_score']}/10，继续改进...")
            topic = self._improve_topic(topic, result['report'])

        logger.warning("❌ 未能达到质量要求")
        return result


    def _improve_topic(self, topic: str, previous_report: str) -> str:
        """
        改进研究主题

        Args:
            topic: 原始主题
            previous_report: 之前的报告

        Returns:
            改进后的主题
        """
        # 简单的改进策略：添加更具体的要求
        improvements = [
            f"{topic}的详细信息和最佳实践",
            f"{topic}的深入分析和案例研究",
            f"{topic}的全面指南和实用建议"
        ]

        # 返回下一个改进版本
        return improvements[0]


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    print("\n" + "="*50)
    print("研究员Agent示例")
    print("="*50 + "\n")

    # 创建模拟搜索工具
    def mock_search(query: str) -> str:
        return f"关于'{query}'的搜索结果：这是一些相关信息..."

    search_tool = Tool(
        name="search",
        func=mock_search,
        description="搜索信息"
    )

    # 创建研究员
    researcher = ResearcherAgent(tools=[search_tool], verbose=False)

    # 示例1：基础研究
    print("示例1: 基础研究")
    result = researcher.research("Python最佳实践")
    print(f"成功: {result['success']}")
    print(f"质量评分: {result['quality_score']}/10")
    print(f"报告长度: {len(result.get('report', ''))}")
    print()

    # 示例2：闭环研究
    print("示例2: 闭环研究")
    result = researcher.research_with_verification("AI Agent设计", min_quality=7.0)
    print(f"成功: {result['success']}")
    print(f"质量评分: {result['quality_score']}/10")


if __name__ == "__main__":
    example_usage()
