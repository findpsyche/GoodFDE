"""
成本效益分析

本示例展示如何：
1. 计算云端 API 成本
2. 计算本地部署成本
3. 对比不同方案的 ROI
4. 生成成本分析报告
5. 提供决策建议

学习目标：
- 理解成本构成
- 掌握成本计算方法
- 学会进行成本效益分析
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

console = Console()


@dataclass
class CloudAPICost:
    """云端 API 成本"""
    provider: str  # OpenAI, Anthropic, etc.
    model: str
    input_price_per_1k: float  # 每 1K tokens 价格
    output_price_per_1k: float
    monthly_requests: int
    avg_input_tokens: int
    avg_output_tokens: int

    @property
    def monthly_cost(self) -> float:
        """月度成本"""
        input_cost = (self.monthly_requests * self.avg_input_tokens / 1000) * self.input_price_per_1k
        output_cost = (self.monthly_requests * self.avg_output_tokens / 1000) * self.output_price_per_1k
        return input_cost + output_cost

    @property
    def cost_per_request(self) -> float:
        """每请求成本"""
        return self.monthly_cost / self.monthly_requests if self.monthly_requests > 0 else 0


@dataclass
class LocalDeploymentCost:
    """本地部署成本"""
    hardware_name: str
    hardware_cost: float  # 硬件成本
    monthly_electricity: float  # 月度电费
    monthly_maintenance: float  # 月度维护成本
    depreciation_months: int = 36  # 折旧月数
    monthly_requests: int = 0

    @property
    def monthly_depreciation(self) -> float:
        """月度折旧"""
        return self.hardware_cost / self.depreciation_months

    @property
    def monthly_total_cost(self) -> float:
        """月度总成本"""
        return self.monthly_depreciation + self.monthly_electricity + self.monthly_maintenance

    @property
    def cost_per_request(self) -> float:
        """每请求成本"""
        return self.monthly_total_cost / self.monthly_requests if self.monthly_requests > 0 else 0


@dataclass
class CostComparison:
    """成本对比"""
    scenario: str
    monthly_requests: int
    cloud_cost: float
    local_cost: float
    breakeven_months: float
    recommendation: str


class CostAnalyzer:
    """成本分析器"""

    def __init__(self):
        self.cloud_providers = self._init_cloud_providers()
        self.hardware_configs = self._init_hardware_configs()

    def _init_cloud_providers(self) -> Dict[str, Dict[str, Any]]:
        """初始化云端 API 价格"""
        return {
            "openai_gpt4": {
                "provider": "OpenAI",
                "model": "GPT-4",
                "input_price": 0.03,  # $0.03 per 1K tokens
                "output_price": 0.06
            },
            "openai_gpt35": {
                "provider": "OpenAI",
                "model": "GPT-3.5 Turbo",
                "input_price": 0.0015,
                "output_price": 0.002
            },
            "anthropic_claude3_opus": {
                "provider": "Anthropic",
                "model": "Claude 3 Opus",
                "input_price": 0.015,
                "output_price": 0.075
            },
            "anthropic_claude3_sonnet": {
                "provider": "Anthropic",
                "model": "Claude 3 Sonnet",
                "input_price": 0.003,
                "output_price": 0.015
            }
        }

    def _init_hardware_configs(self) -> Dict[str, Dict[str, Any]]:
        """初始化硬件配置"""
        return {
            "rtx_3090": {
                "name": "RTX 3090 (24GB)",
                "cost": 1500,
                "power_watts": 350,
                "suitable_models": ["7B-13B"]
            },
            "rtx_4090": {
                "name": "RTX 4090 (24GB)",
                "cost": 2000,
                "power_watts": 450,
                "suitable_models": ["7B-13B"]
            },
            "a100_40gb": {
                "name": "A100 (40GB)",
                "cost": 10000,
                "power_watts": 400,
                "suitable_models": ["7B-70B"]
            },
            "a100_80gb": {
                "name": "A100 (80GB)",
                "cost": 15000,
                "power_watts": 400,
                "suitable_models": ["7B-70B+"]
            }
        }

    def calculate_electricity_cost(
        self,
        power_watts: int,
        hours_per_day: int = 24,
        price_per_kwh: float = 0.12
    ) -> float:
        """计算月度电费"""
        kwh_per_month = (power_watts / 1000) * hours_per_day * 30
        return kwh_per_month * price_per_kwh

    def calculate_breakeven(
        self,
        hardware_cost: float,
        monthly_cloud_cost: float,
        monthly_local_cost: float
    ) -> float:
        """计算回本月数"""
        if monthly_cloud_cost <= monthly_local_cost:
            return float('inf')  # 永远不回本

        monthly_savings = monthly_cloud_cost - monthly_local_cost
        return hardware_cost / monthly_savings if monthly_savings > 0 else float('inf')

    def compare_scenarios(
        self,
        monthly_requests: int,
        avg_input_tokens: int = 500,
        avg_output_tokens: int = 500
    ) -> List[CostComparison]:
        """对比不同场景"""
        comparisons = []

        # 云端方案
        cloud_api = CloudAPICost(
            provider="OpenAI",
            model="GPT-4",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
            monthly_requests=monthly_requests,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens
        )

        # 本地方案
        local_deploy = LocalDeploymentCost(
            hardware_name="RTX 4090",
            hardware_cost=2000,
            monthly_electricity=self.calculate_electricity_cost(450),
            monthly_maintenance=50,
            monthly_requests=monthly_requests
        )

        breakeven = self.calculate_breakeven(
            local_deploy.hardware_cost,
            cloud_api.monthly_cost,
            local_deploy.monthly_total_cost
        )

        # 决策建议
        if monthly_requests < 1000:
            recommendation = "推荐云端 API - 使用量低，无需硬件投入"
        elif breakeven < 6:
            recommendation = "推荐本地部署 - 6个月内回本，长期成本低"
        elif breakeven < 12:
            recommendation = "考虑本地部署 - 1年内回本，适合长期使用"
        else:
            recommendation = "推荐云端 API - 回本周期长，不划算"

        comparisons.append(CostComparison(
            scenario=f"{monthly_requests} 请求/月",
            monthly_requests=monthly_requests,
            cloud_cost=cloud_api.monthly_cost,
            local_cost=local_deploy.monthly_total_cost,
            breakeven_months=breakeven,
            recommendation=recommendation
        ))

        return comparisons


def demo_cloud_api_costs():
    """演示云端 API 成本"""
    console.print("\n[bold cyan]🚀 演示1: 云端 API 成本分析[/bold cyan]\n")

    analyzer = CostAnalyzer()

    console.print("[yellow]主流云端 API 价格 (2024):[/yellow]\n")

    pricing_table = Table(show_header=True, header_style="bold magenta")
    pricing_table.add_column("提供商", style="cyan")
    pricing_table.add_column("模型")
    pricing_table.add_column("输入价格", justify="right")
    pricing_table.add_column("输出价格", justify="right")

    for key, info in analyzer.cloud_providers.items():
        pricing_table.add_row(
            info["provider"],
            info["model"],
            f"${info['input_price']}/1K",
            f"${info['output_price']}/1K"
        )

    console.print(pricing_table)

    console.print("\n[yellow]成本计算示例:[/yellow]\n")

    # 示例场景
    scenarios = [
        {"requests": 1000, "input": 500, "output": 500},
        {"requests": 10000, "input": 500, "output": 500},
        {"requests": 100000, "input": 500, "output": 500}
    ]

    for scenario in scenarios:
        console.print(f"[bold]场景: {scenario['requests']} 请求/月[/bold]")
        console.print(f"[dim]平均输入: {scenario['input']} tokens, 输出: {scenario['output']} tokens[/dim]\n")

        for key, info in list(analyzer.cloud_providers.items())[:2]:  # 只显示前两个
            cost = CloudAPICost(
                provider=info["provider"],
                model=info["model"],
                input_price_per_1k=info["input_price"],
                output_price_per_1k=info["output_price"],
                monthly_requests=scenario["requests"],
                avg_input_tokens=scenario["input"],
                avg_output_tokens=scenario["output"]
            )

            console.print(f"  {info['model']}: ${cost.monthly_cost:.2f}/月 (${cost.cost_per_request:.4f}/请求)")

        console.print()


def demo_local_deployment_costs():
    """演示本地部署成本"""
    console.print("\n[bold cyan]🚀 演示2: 本地部署成本分析[/bold cyan]\n")

    analyzer = CostAnalyzer()

    console.print("[yellow]硬件配置和成本:[/yellow]\n")

    hardware_table = Table(show_header=True, header_style="bold magenta")
    hardware_table.add_column("硬件", style="cyan")
    hardware_table.add_column("价格", justify="right")
    hardware_table.add_column("功耗", justify="right")
    hardware_table.add_column("月度电费", justify="right")
    hardware_table.add_column("适用模型")

    for key, hw in analyzer.hardware_configs.items():
        electricity = analyzer.calculate_electricity_cost(hw["power_watts"])
        hardware_table.add_row(
            hw["name"],
            f"${hw['cost']}",
            f"{hw['power_watts']}W",
            f"${electricity:.2f}",
            ", ".join(hw["suitable_models"])
        )

    console.print(hardware_table)

    console.print("\n[yellow]成本构成示例 (RTX 4090):[/yellow]\n")

    local_cost = LocalDeploymentCost(
        hardware_name="RTX 4090",
        hardware_cost=2000,
        monthly_electricity=analyzer.calculate_electricity_cost(450),
        monthly_maintenance=50,
        monthly_requests=10000
    )

    cost_breakdown = Table(show_header=True, header_style="bold magenta")
    cost_breakdown.add_column("成本项", style="cyan")
    cost_breakdown.add_column("金额", justify="right", style="green")
    cost_breakdown.add_column("说明")

    cost_breakdown.add_row(
        "硬件成本",
        f"${local_cost.hardware_cost}",
        "一次性投入"
    )
    cost_breakdown.add_row(
        "月度折旧",
        f"${local_cost.monthly_depreciation:.2f}",
        f"3年折旧"
    )
    cost_breakdown.add_row(
        "月度电费",
        f"${local_cost.monthly_electricity:.2f}",
        "24小时运行"
    )
    cost_breakdown.add_row(
        "月度维护",
        f"${local_cost.monthly_maintenance:.2f}",
        "人工、网络等"
    )
    cost_breakdown.add_row(
        "月度总成本",
        f"${local_cost.monthly_total_cost:.2f}",
        "所有成本之和"
    )
    cost_breakdown.add_row(
        "每请求成本",
        f"${local_cost.cost_per_request:.4f}",
        f"基于 {local_cost.monthly_requests} 请求/月"
    )

    console.print(cost_breakdown)


def demo_cost_comparison():
    """演示成本对比"""
    console.print("\n[bold cyan]🚀 演示3: 云端 vs 本地成本对比[/bold cyan]\n")

    analyzer = CostAnalyzer()

    # 不同使用量场景
    request_volumes = [1000, 5000, 10000, 50000, 100000]

    console.print("[yellow]不同使用量下的成本对比:[/yellow]\n")

    comparison_table = Table(show_header=True, header_style="bold magenta")
    comparison_table.add_column("月请求数", justify="right")
    comparison_table.add_column("云端成本", justify="right")
    comparison_table.add_column("本地成本", justify="right")
    comparison_table.add_column("月度节省", justify="right")
    comparison_table.add_column("回本月数", justify="right")

    for requests in request_volumes:
        comparisons = analyzer.compare_scenarios(requests)

        if comparisons:
            comp = comparisons[0]
            savings = comp.cloud_cost - comp.local_cost

            comparison_table.add_row(
                f"{requests:,}",
                f"${comp.cloud_cost:.2f}",
                f"${comp.local_cost:.2f}",
                f"${savings:.2f}" if savings > 0 else f"-${abs(savings):.2f}",
                f"{comp.breakeven_months:.1f}" if comp.breakeven_months != float('inf') else "∞"
            )

    console.print(comparison_table)


def demo_roi_analysis():
    """演示 ROI 分析"""
    console.print("\n[bold cyan]🚀 演示4: ROI (投资回报率) 分析[/bold cyan]\n")

    analyzer = CostAnalyzer()

    # 场景：月请求 10,000
    monthly_requests = 10000

    cloud_cost = CloudAPICost(
        provider="OpenAI",
        model="GPT-4",
        input_price_per_1k=0.03,
        output_price_per_1k=0.06,
        monthly_requests=monthly_requests,
        avg_input_tokens=500,
        avg_output_tokens=500
    )

    local_cost = LocalDeploymentCost(
        hardware_name="RTX 4090",
        hardware_cost=2000,
        monthly_electricity=analyzer.calculate_electricity_cost(450),
        monthly_maintenance=50,
        monthly_requests=monthly_requests
    )

    console.print(f"[yellow]场景: {monthly_requests:,} 请求/月[/yellow]\n")

    # 计算多年成本
    years = [1, 2, 3]
    roi_table = Table(show_header=True, header_style="bold magenta")
    roi_table.add_column("时间", style="cyan")
    roi_table.add_column("云端累计", justify="right")
    roi_table.add_column("本地累计", justify="right")
    roi_table.add_column("节省", justify="right")
    roi_table.add_column("ROI", justify="right")

    for year in years:
        months = year * 12
        cloud_total = cloud_cost.monthly_cost * months
        local_total = local_cost.hardware_cost + (local_cost.monthly_total_cost * months)
        savings = cloud_total - local_total
        roi = (savings / local_cost.hardware_cost) * 100 if local_cost.hardware_cost > 0 else 0

        roi_table.add_row(
            f"{year} 年",
            f"${cloud_total:,.2f}",
            f"${local_total:,.2f}",
            f"${savings:,.2f}",
            f"{roi:.1f}%"
        )

    console.print(roi_table)

    console.print("\n[yellow]分析:[/yellow]")
    breakeven = analyzer.calculate_breakeven(
        local_cost.hardware_cost,
        cloud_cost.monthly_cost,
        local_cost.monthly_total_cost
    )

    if breakeven != float('inf'):
        console.print(f"• 回本周期: {breakeven:.1f} 个月")
        console.print(f"• 第一年 ROI: {((cloud_cost.monthly_cost * 12 - local_cost.hardware_cost - local_cost.monthly_total_cost * 12) / local_cost.hardware_cost * 100):.1f}%")
        console.print(f"• 三年总节省: ${(cloud_cost.monthly_cost * 36 - local_cost.hardware_cost - local_cost.monthly_total_cost * 36):,.2f}")
    else:
        console.print("• 云端方案更经济，不建议本地部署")


def demo_decision_matrix():
    """演示决策矩阵"""
    console.print("\n[bold cyan]🚀 演示5: 决策矩阵[/bold cyan]\n")

    console.print("[yellow]选择云端 API 的场景:[/yellow]\n")

    cloud_scenarios = [
        ("✅ 低频使用", "< 1,000 请求/月"),
        ("✅ 快速上线", "无需等待硬件采购"),
        ("✅ 弹性需求", "使用量波动大"),
        ("✅ 无运维能力", "没有专业运维团队"),
        ("✅ 多模型需求", "需要频繁切换模型"),
        ("✅ 最新模型", "需要使用最新的模型")
    ]

    for scenario, desc in cloud_scenarios:
        console.print(f"{scenario}: {desc}")

    console.print("\n[yellow]选择本地部署的场景:[/yellow]\n")

    local_scenarios = [
        ("✅ 高频使用", "> 10,000 请求/月"),
        ("✅ 数据隐私", "敏感数据不能上云"),
        ("✅ 成本敏感", "长期使用，关注成本"),
        ("✅ 低延迟", "需要极低的响应延迟"),
        ("✅ 离线使用", "无网络或网络不稳定"),
        ("✅ 定制需求", "需要微调或定制模型")
    ]

    for scenario, desc in local_scenarios:
        console.print(f"{scenario}: {desc}")

    console.print("\n[yellow]混合方案:[/yellow]\n")

    hybrid_scenarios = [
        "• 常见任务用本地模型（成本低）",
        "• 复杂任务用云端 API（质量高）",
        "• 高峰期用云端 API（弹性扩展）",
        "• 平时用本地模型（节省成本）"
    ]

    for scenario in hybrid_scenarios:
        console.print(scenario)


def demo_report_generation():
    """演示生成成本分析报告"""
    console.print("\n[bold cyan]🚀 演示6: 生成成本分析报告[/bold cyan]\n")

    analyzer = CostAnalyzer()

    # 分析多个场景
    scenarios = [1000, 10000, 50000]
    all_comparisons = []

    for requests in scenarios:
        comparisons = analyzer.compare_scenarios(requests)
        all_comparisons.extend(comparisons)

    # 生成报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "cost_comparison",
        "scenarios": [asdict(comp) for comp in all_comparisons],
        "recommendations": {
            "low_volume": "使用云端 API",
            "medium_volume": "考虑本地部署",
            "high_volume": "强烈推荐本地部署"
        }
    }

    # 保存报告
    report_file = f"cost_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✅ 报告已保存: {report_file}[/green]")

    # 显示摘要
    console.print("\n[bold]📄 报告摘要:[/bold]\n")

    for comp in all_comparisons:
        console.print(f"[yellow]{comp.scenario}:[/yellow]")
        console.print(f"  云端: ${comp.cloud_cost:.2f}/月")
        console.print(f"  本地: ${comp.local_cost:.2f}/月")
        console.print(f"  回本: {comp.breakeven_months:.1f} 个月")
        console.print(f"  建议: {comp.recommendation}\n")


def main():
    """主函数"""
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]成本效益分析[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    demos = [
        ("云端 API 成本", demo_cloud_api_costs),
        ("本地部署成本", demo_local_deployment_costs),
        ("成本对比", demo_cost_comparison),
        ("ROI 分析", demo_roi_analysis),
        ("决策矩阵", demo_decision_matrix),
        ("生成报告", demo_report_generation)
    ]

    console.print("\n[bold]选择要运行的演示:[/bold]")
    for i, (name, _) in enumerate(demos, 1):
        console.print(f"  {i}. {name}")
    console.print("  0. 运行所有演示")

    choice = input("\n请输入选项 (0-6): ").strip()

    if choice == "0":
        for name, func in demos:
            func()
    elif choice.isdigit() and 1 <= int(choice) <= len(demos):
        demos[int(choice) - 1][1]()
    else:
        console.print("[red]❌ 无效选项[/red]")


if __name__ == "__main__":
    main()
