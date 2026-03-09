"""
协调员Agent

职责：
- 协调多个Agent
- 分配任务
- 汇总结果

设计原则：
- 验证驱动：验证每个Agent的输出
- 闭环思维：迭代改进直到满足要求
- 降低认知负荷：清晰的协调流程
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class CoordinatorAgent:
    """
    协调员Agent

    使用场景：
    - 需要协调多个Agent
    - 需要管理复杂工作流
    - 需要确保整体质量

    设计层次：
    - ✅ 及格线：能分配、能汇总
    - 🌟 状元：能迭代、能优化
    """

    def __init__(self, verbose: bool = True):
        """
        初始化协调员Agent

        Args:
            verbose: 是否显示详细日志
        """
        self.verbose = verbose
        self.agents = {}

        logger.info("✅ 协调员Agent初始化完成")

    def register_agent(self, name: str, agent: Any):
        """
        注册Agent

        Args:
            name: Agent名称
            agent: Agent实例

        示例:
            >>> coordinator = CoordinatorAgent()
            >>> coordinator.register_agent('researcher', researcher_agent)
        """
        self.agents[name] = agent
        logger.info(f"✅ 注册Agent: {name}")

    def coordinate(self, task: str, workflow: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        协调执行任务

        Args:
            task: 任务描述
            workflow: 工作流定义

        Returns:
            执行结果

        工作流格式:
            [
                {
                    'agent': 'researcher',
                    'method': 'research',
                    'args': {'topic': task}
                },
                {
                    'agent': 'editor',
                    'method': 'edit',
                    'args': {'content': '{{researcher.output}}'}
                }
            ]

        示例:
            >>> workflow = [
            ...     {'agent': 'researcher', 'method': 'research', 'args': {'topic': task}},
            ...     {'agent': 'editor', 'method': 'edit', 'args': {'content': '{{researcher.output}}'}}
            ... ]
            >>> result = coordinator.coordinate("Python最佳实践", workflow)
        """
        logger.info(f"🎯 开始协调任务: {task}")

        results = {}
        context = {'task': task}

        try:
            for i, step in enumerate(workflow, 1):
                agent_name = step['agent']
                method_name = step['method']
                args = step.get('args', {})

                logger.info(f"📍 步骤 {i}/{len(workflow)}: {agent_name}.{method_name}")

                # 获取Agent
                if agent_name not in self.agents:
                    raise ValueError(f"未注册的Agent: {agent_name}")

                agent = self.agents[agent_name]

                # 解析参数（替换占位符）
                resolved_args = self._resolve_args(args, context)

                # 执行方法
                method = getattr(agent, method_name)
                result = method(**resolved_args)

                # 保存结果
                results[agent_name] = result
                context[agent_name] = result

                # 验证结果
                if not result.get('success', True):
                    logger.warning(f"⚠️  步骤失败: {agent_name}")
                    return {
                        'success': False,
                        'failed_at': agent_name,
                        'error': result.get('error', 'Unknown error'),
                        'results': results
                    }

            logger.info("✅ 任务协调完成")

            return {
                'success': True,
                'results': results,
                'final_output': self._extract_final_output(results, workflow)
            }

        except Exception as e:
            logger.error(f"❌ 协调失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': results
            }

    def coordinate_with_iteration(
        self,
        task: str,
        workflow: List[Dict[str, Any]],
        max_iterations: int = 3,
        quality_threshold: float = 8.0
    ) -> Dict[str, Any]:
        """
        带迭代的协调（闭环模式）

        Args:
            task: 任务描述
            workflow: 工作流定义
            max_iterations: 最大迭代次数
            quality_threshold: 质量阈值

        Returns:
            执行结果
        """
        logger.info(f"🔄 开始迭代协调: {task} (最大迭代: {max_iterations})")

        for iteration in range(max_iterations):
            logger.info(f"\n{'='*50}")
            logger.info(f"迭代 {iteration + 1}/{max_iterations}")
            logger.info(f"{'='*50}\n")

            # 执行工作流
            result = self.coordinate(task, workflow)

            if not result['success']:
                logger.warning(f"迭代 {iteration + 1} 失败")
                continue

            # 评估质量
            quality_score = self._evaluate_quality(result)

            logger.info(f"质量评分: {quality_score:.1f}/10")

            # 检查是否达标
            if quality_score >= quality_threshold:
                logger.info(f"✅ 质量达标！")
                result['iterations'] = iteration + 1
                result['quality_score'] = quality_score
                return result

            # 生成改进建议
            if iteration < max_iterations - 1:
                feedback = self._generate_feedback(result, quality_score)
                logger.info(f"💡 改进建议: {feedback}")

                # 更新任务（添加反馈）
                task = f"{task}\n\n改进要求：{feedback}"

        logger.warning("❌ 未能在最大迭代次数内达到质量要求")
        result['iterations'] = max_iterations
        result['quality_score'] = quality_score
        return result

    def _resolve_args(self, args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析参数中的占位符

        Args:
            args: 参数字典
            context: 上下文

        Returns:
            解析后的参数
        """
        resolved = {}

        for key, value in args.items():
            if isinstance(value, str) and '{{' in value and '}}' in value:
                # 提取占位符
                placeholder = value.strip('{}').strip()
                parts = placeholder.split('.')

                # 从上下文中获取值
                obj = context
                for part in parts:
                    if isinstance(obj, dict):
                        obj = obj.get(part, value)
                    else:
                        obj = getattr(obj, part, value)

                resolved[key] = obj
            else:
                resolved[key] = value

        return resolved

    def _extract_final_output(self, results: Dict[str, Any], workflow: List[Dict[str, Any]]) -> Any:
        """提取最终输出"""
        # 返回最后一个Agent的输出
        if workflow:
            last_agent = workflow[-1]['agent']
            return results.get(last_agent)
        return None

    def _evaluate_quality(self, result: Dict[str, Any]) -> float:
        """
        评估整体质量

        Args:
            result: 执行结果

        Returns:
            质量评分（0-10）
        """
        # 简单的质量评估
        score = 5.0

        # 检查是否成功
        if result.get('success'):
            score += 2.0

        # 检查各个Agent的评分
        results = result.get('results', {})

        for agent_name, agent_result in results.items():
            if isinstance(agent_result, dict):
                # 检查质量评分
                if 'quality_score' in agent_result:
                    score += agent_result['quality_score'] * 0.1

                # 检查改进评分
                if 'improvement_score' in agent_result:
                    score += agent_result['improvement_score'] * 0.1

                # 检查审核评分
                if 'overall_score' in agent_result:
                    score += agent_result['overall_score'] * 0.1

        return min(score, 10.0)

    def _generate_feedback(self, result: Dict[str, Any], quality_score: float) -> str:
        """
        生成改进反馈

        Args:
            result: 执行结果
            quality_score: 质量评分

        Returns:
            反馈建议
        """
        feedback_items = []

        # 根据质量评分生成反馈
        if quality_score < 6.0:
            feedback_items.append("内容需要大幅改进，请增加深度和广度")
        elif quality_score < 8.0:
            feedback_items.append("内容基本合格，但需要进一步优化")

        # 检查审核反馈
        results = result.get('results', {})
        if 'reviewer' in results:
            reviewer_result = results['reviewer']
            if isinstance(reviewer_result, dict):
                agent_feedback = reviewer_result.get('feedback', [])
                feedback_items.extend(agent_feedback[:3])  # 最多3条

        return '; '.join(feedback_items) if feedback_items else "继续改进内容质量"


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    print("\n" + "="*50)
    print("协调员Agent示例")
    print("="*50 + "\n")

    # 创建模拟Agent
    class MockAgent:
        def process(self, input_data):
            return {
                'success': True,
                'output': f"处理结果: {input_data}",
                'quality_score': 8.0
            }

    coordinator = CoordinatorAgent(verbose=True)

    # 注册Agent
    coordinator.register_agent('agent1', MockAgent())
    coordinator.register_agent('agent2', MockAgent())

    # 定义工作流
    workflow = [
        {
            'agent': 'agent1',
            'method': 'process',
            'args': {'input_data': 'task'}
        },
        {
            'agent': 'agent2',
            'method': 'process',
            'args': {'input_data': '{{agent1.output}}'}
        }
    ]

    # 执行协调
    result = coordinator.coordinate("测试任务", workflow)
    print(f"\n成功: {result['success']}")
    print(f"步骤数: {len(result.get('results', {}))}")


if __name__ == "__main__":
    example_usage()
