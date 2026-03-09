"""
Plan-and-Execute模式示例

Plan-and-Execute是一种先规划后执行的Agent设计模式。
Agent首先制定完整的执行计划，然后按计划逐步执行。

核心特点：
1. 分离规划和执行
2. 计划可以动态调整
3. 适合复杂的多步骤任务

设计原则：
- 验证驱动：验证每一步的执行结果
- 闭环思维：根据执行结果调整计划
- 降低认知负荷：清晰的计划结构
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


# ============================================================================
# 工具定义
# ============================================================================

def search_web(query: str) -> str:
    """模拟网页搜索"""
    logger.info(f"🔍 搜索: {query}")
    return f"关于'{query}'的搜索结果：这是相关信息..."


def calculate(expression: str) -> str:
    """计算数学表达式"""
    logger.info(f"🔢 计算: {expression}")
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


def save_to_file(content: str, filename: str = "output.txt") -> str:
    """保存内容到文件"""
    logger.info(f"💾 保存到文件: {filename}")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"成功保存到 {filename}"
    except Exception as e:
        return f"保存失败：{str(e)}"


tools = [
    Tool(name="search", func=search_web, description="搜索网页信息"),
    Tool(name="calculate", func=calculate, description="计算数学表达式"),
    Tool(name="save", func=save_to_file, description="保存内容到文件")
]


# ============================================================================
# Plan-and-Execute Agent
# ============================================================================

class PlanAndExecuteAgent:
    """
    Plan-and-Execute Agent

    工作流程：
    1. Planner：制定执行计划
    2. Executor：按计划执行每一步
    3. Replanner：根据结果调整计划（可选）
    """

    def __init__(self, tools: List[Tool], llm: Optional[Any] = None, verbose: bool = True):
        """
        初始化Agent

        Args:
            tools: 可用工具列表
            llm: 语言模型
            verbose: 是否显示详细日志
        """
        self.tools = tools
        self.tool_dict = {tool.name: tool for tool in tools}
        self.llm = llm or self._get_default_llm()
        self.verbose = verbose

        # 创建Planner和Executor
        self.planner = self._create_planner()
        self.executor = self._create_executor()

        logger.info("✅ Plan-and-Execute Agent初始化完成")

    def _get_default_llm(self):
        """获取默认LLM"""
        if os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
        elif os.getenv("ANTHROPIC_API_KEY"):
            return ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
        else:
            raise ValueError("请设置OPENAI_API_KEY或ANTHROPIC_API_KEY")

    def _create_planner(self) -> LLMChain:
        """创建Planner"""
        prompt = PromptTemplate.from_template("""
你是一个专业的任务规划师，需要为给定任务制定详细的执行计划。

可用工具：
{tools}

任务：{task}

请制定一个详细的执行计划，格式如下：

## 执行计划

### 步骤1: [步骤描述]
- 工具: [工具名称]
- 输入: [工具输入]
- 预期输出: [预期结果]

### 步骤2: [步骤描述]
- 工具: [工具名称]
- 输入: [工具输入]
- 预期输出: [预期结果]

...

### 最终目标
[描述最终要达成的目标]

注意：
1. 每个步骤要清晰、具体
2. 步骤之间要有逻辑关系
3. 考虑可能的错误情况
4. 确保能达成最终目标

执行计划：
""")

        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    def _create_executor(self) -> LLMChain:
        """创建Executor"""
        prompt = PromptTemplate.from_template("""
你是一个任务执行器，需要执行给定的步骤。

当前步骤：
{step}

可用工具：
{tools}

之前的执行结果：
{previous_results}

请执行这个步骤，并返回：
1. 使用的工具名称
2. 工具的输入参数
3. 执行结果

格式：
工具: [工具名称]
输入: [输入参数]
结果: [执行结果]

执行：
""")

        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    def run(self, task: str, max_steps: int = 10) -> Dict[str, Any]:
        """
        执行任务

        Args:
            task: 任务描述
            max_steps: 最大执行步骤数

        Returns:
            执行结果
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"开始执行任务: {task}")
        logger.info(f"{'='*70}\n")

        try:
            # 阶段1：制定计划
            logger.info("📋 阶段1: 制定执行计划")
            plan = self._make_plan(task)
            logger.info(f"\n执行计划:\n{plan}\n")

            # 阶段2：执行计划
            logger.info("⚙️  阶段2: 执行计划")
            results = self._execute_plan(plan, max_steps)

            # 阶段3：总结结果
            logger.info("📊 阶段3: 总结结果")
            summary = self._summarize_results(task, results)

            logger.info(f"\n{'='*70}")
            logger.info("✅ 任务执行完成")
            logger.info(f"{'='*70}\n")

            return {
                'success': True,
                'task': task,
                'plan': plan,
                'results': results,
                'summary': summary,
                'steps_executed': len(results)
            }

        except Exception as e:
            logger.error(f"❌ 任务执行失败: {e}")
            return {
                'success': False,
                'task': task,
                'error': str(e)
            }

    def _make_plan(self, task: str) -> str:
        """制定计划"""
        tool_descriptions = '\n'.join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])

        plan = self.planner.run(
            task=task,
            tools=tool_descriptions
        )

        return plan

    def _execute_plan(self, plan: str, max_steps: int) -> List[Dict[str, Any]]:
        """执行计划"""
        # 解析计划中的步骤
        steps = self._parse_plan(plan)

        results = []
        previous_results = []

        for i, step in enumerate(steps[:max_steps], 1):
            logger.info(f"\n{'─'*70}")
            logger.info(f"执行步骤 {i}/{len(steps)}: {step['description']}")
            logger.info(f"{'─'*70}")

            try:
                # 执行步骤
                result = self._execute_step(step, previous_results)

                results.append({
                    'step_number': i,
                    'description': step['description'],
                    'tool': result.get('tool'),
                    'input': result.get('input'),
                    'output': result.get('output'),
                    'success': result.get('success', True)
                })

                previous_results.append(result)

                # 验证结果
                if not result.get('success', True):
                    logger.warning(f"⚠️  步骤 {i} 执行失败，继续下一步")

                logger.info(f"✅ 步骤 {i} 完成")

            except Exception as e:
                logger.error(f"❌ 步骤 {i} 执行错误: {e}")
                results.append({
                    'step_number': i,
                    'description': step['description'],
                    'error': str(e),
                    'success': False
                })

        return results

    def _parse_plan(self, plan: str) -> List[Dict[str, Any]]:
        """
        解析计划

        Args:
            plan: 计划文本

        Returns:
            步骤列表
        """
        steps = []
        lines = plan.split('\n')

        current_step = None

        for line in lines:
            line = line.strip()

            # 检测步骤标题
            if line.startswith('### 步骤') or line.startswith('##步骤'):
                if current_step:
                    steps.append(current_step)

                # 提取步骤描述
                description = line.split(':', 1)[1].strip() if ':' in line else line
                current_step = {
                    'description': description,
                    'tool': None,
                    'input': None
                }

            # 提取工具和输入
            elif current_step:
                if line.startswith('- 工具:') or line.startswith('-工具:'):
                    current_step['tool'] = line.split(':', 1)[1].strip()
                elif line.startswith('- 输入:') or line.startswith('-输入:'):
                    current_step['input'] = line.split(':', 1)[1].strip()

        # 添加最后一个步骤
        if current_step:
            steps.append(current_step)

        # 如果没有解析到步骤，创建默认步骤
        if not steps:
            steps = [
                {
                    'description': '执行任务',
                    'tool': 'search',
                    'input': '相关信息'
                }
            ]

        return steps

    def _execute_step(self, step: Dict[str, Any], previous_results: List[Dict]) -> Dict[str, Any]:
        """执行单个步骤"""
        tool_name = step.get('tool')
        tool_input = step.get('input', '')

        # 如果没有指定工具，尝试从描述中推断
        if not tool_name:
            tool_name = self._infer_tool(step['description'])

        # 获取工具
        if tool_name not in self.tool_dict:
            return {
                'success': False,
                'error': f"工具不存在: {tool_name}"
            }

        tool = self.tool_dict[tool_name]

        try:
            # 执行工具
            logger.info(f"🔧 使用工具: {tool_name}")
            logger.info(f"📥 输入: {tool_input}")

            output = tool.func(tool_input)

            logger.info(f"📤 输出: {output[:100]}...")

            return {
                'success': True,
                'tool': tool_name,
                'input': tool_input,
                'output': output
            }

        except Exception as e:
            logger.error(f"❌ 工具执行失败: {e}")
            return {
                'success': False,
                'tool': tool_name,
                'input': tool_input,
                'error': str(e)
            }

    def _infer_tool(self, description: str) -> str:
        """从描述中推断工具"""
        description_lower = description.lower()

        if '搜索' in description_lower or 'search' in description_lower:
            return 'search'
        elif '计算' in description_lower or 'calculate' in description_lower:
            return 'calculate'
        elif '保存' in description_lower or 'save' in description_lower:
            return 'save'

        # 默认使用第一个工具
        return self.tools[0].name if self.tools else None

    def _summarize_results(self, task: str, results: List[Dict]) -> str:
        """总结执行结果"""
        successful_steps = sum(1 for r in results if r.get('success', False))
        total_steps = len(results)

        summary = f"""
## 任务执行总结

**任务**: {task}

**执行情况**:
- 总步骤数: {total_steps}
- 成功步骤: {successful_steps}
- 失败步骤: {total_steps - successful_steps}
- 成功率: {successful_steps/total_steps*100:.1f}%

**详细结果**:
"""

        for result in results:
            status = "✅" if result.get('success', False) else "❌"
            summary += f"\n{status} 步骤{result['step_number']}: {result['description']}"

        return summary


# ============================================================================
# 示例函数
# ============================================================================

def example_1_basic():
    """示例1：基础Plan-and-Execute"""
    print("\n" + "="*70)
    print("示例1: Plan-and-Execute模式 - 基础")
    print("="*70 + "\n")

    agent = PlanAndExecuteAgent(tools=tools, verbose=False)

    task = "研究Python最佳实践，总结要点，并保存到文件"

    result = agent.run(task)

    if result['success']:
        print("\n" + "-"*70)
        print("执行总结:")
        print(result['summary'])
        print("-"*70)


def example_2_complex():
    """示例2：复杂任务"""
    print("\n" + "="*70)
    print("示例2: Plan-and-Execute模式 - 复杂任务")
    print("="*70 + "\n")

    agent = PlanAndExecuteAgent(tools=tools, verbose=False)

    task = """
    完成以下任务：
    1. 搜索关于AI Agent的信息
    2. 计算如果开发一个Agent需要100小时，每天工作8小时，需要多少天
    3. 将结果保存到文件
    """

    result = agent.run(task)

    if result['success']:
        print(f"\n执行了 {result['steps_executed']} 个步骤")
        print(f"计划:\n{result['plan'][:200]}...")


def example_3_with_verification():
    """示例3：带验证的执行"""
    print("\n" + "="*70)
    print("示例3: Plan-and-Execute模式 - 带验证")
    print("="*70 + "\n")

    agent = PlanAndExecuteAgent(tools=tools, verbose=False)

    task = "搜索Python教程，并验证搜索结果"

    result = agent.run(task, max_steps=5)

    if result['success']:
        # 验证每个步骤
        print("\n步骤验证:")
        for step_result in result['results']:
            status = "✅ 通过" if step_result.get('success') else "❌ 失败"
            print(f"  步骤{step_result['step_number']}: {status}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("\n" + "="*70)
    print("Plan-and-Execute模式示例")
    print("="*70)
    print("\n核心特点：")
    print("  ✅ 分离规划和执行")
    print("  ✅ 计划可以动态调整")
    print("  ✅ 适合复杂的多步骤任务")
    print("\n" + "="*70 + "\n")

    # 检查API密钥
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("❌ 错误：请设置API密钥")
        return

    try:
        # 运行示例
        example_1_basic()

        input("\n按Enter继续下一个示例...")
        example_2_complex()

        input("\n按Enter继续下一个示例...")
        example_3_with_verification()

        print("\n" + "="*70)
        print("✅ 所有示例运行完成！")
        print("="*70)
        print("\n💡 学习要点：")
        print("  1. Plan-and-Execute先制定计划，再执行")
        print("  2. 计划可以包含多个步骤")
        print("  3. 每个步骤都可以验证")
        print("  4. 适合需要明确步骤的任务")
        print("\n📝 对比ReAct模式：")
        print("  - ReAct: 边思考边行动，更灵活")
        print("  - Plan-and-Execute: 先规划后执行，更结构化")
        print("\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)
        print(f"\n❌ 程序执行失败: {e}")


if __name__ == "__main__":
    main()
