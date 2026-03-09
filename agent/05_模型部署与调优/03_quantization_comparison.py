"""
量化方法对比示例

本示例展示如何：
1. 理解不同量化方法的原理
2. 对比 GGUF、GPTQ、AWQ 的性能
3. 测试不同量化级别的效果
4. 评估量化对质量的影响
5. 选择最优的量化配置

学习目标：
- 理解量化的权衡（大小 vs 速度 vs 质量）
- 掌握量化方法的选择标准
- 学会评估量化效果

量化格式对比：
- GGUF: 通用格式，支持 CPU，适合 Ollama
- GPTQ: GPU 优化，4-bit 量化，速度快
- AWQ: 激活感知量化，精度最高
"""

import os
import time
import requests
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()

console = Console()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


@dataclass
class QuantizationResult:
    """量化测试结果"""
    model_name: str
    quantization: str
    size_gb: float
    speed_tokens_per_sec: float
    latency_sec: float
    quality_score: float
    response: str


class QuantizationTester:
    """量化测试工具"""

    def __init__(self, ollama_host: str = OLLAMA_HOST):
        self.ollama_host = ollama_host
        self.api_base = f"{ollama_host}/api"

    def list_models(self) -> List[Dict[str, Any]]:
        """列出已安装的模型"""
        try:
            response = requests.get(f"{self.api_base}/tags")
            if response.status_code == 200:
                return response.json().get("models", [])
            return []
        except Exception as e:
            console.print(f"[red]❌ 获取模型列表失败: {e}[/red]")
            return []

    def test_model(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 100
    ) -> Optional[Dict[str, Any]]:
        """测试单个模型"""
        try:
            data = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens
                }
            }

            start_time = time.time()
            response = requests.post(
                f"{self.api_base}/generate",
                json=data,
                timeout=300
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                return {
                    "response": result.get("response", ""),
                    "elapsed": elapsed,
                    "eval_count": result.get("eval_count", 0),
                    "eval_duration": result.get("eval_duration", 0),
                    "total_duration": result.get("total_duration", 0)
                }
            return None

        except Exception as e:
            console.print(f"[red]❌ 测试失败: {e}[/red]")
            return None

    def evaluate_quality(
        self,
        response: str,
        expected_keywords: List[str]
    ) -> float:
        """简单的质量评估（基于关键词）"""
        if not response:
            return 0.0

        response_lower = response.lower()
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)

        return matches / len(expected_keywords) if expected_keywords else 0.5


def demo_quantization_levels():
    """演示不同量化级别"""
    console.print("\n[bold cyan]🚀 演示1: 量化级别对比[/bold cyan]\n")

    console.print("[yellow]量化级别说明:[/yellow]")
    console.print("[dim]Q2_K: 2-bit, 最小, 最快, 质量较差[/dim]")
    console.print("[dim]Q3_K_M: 3-bit, 小, 快, 质量一般[/dim]")
    console.print("[dim]Q4_K_M: 4-bit, 中, 平衡, 质量好 (推荐)[/dim]")
    console.print("[dim]Q5_K_M: 5-bit, 较大, 较慢, 质量很好[/dim]")
    console.print("[dim]Q6_K: 6-bit, 大, 慢, 质量优秀[/dim]")
    console.print("[dim]Q8_0: 8-bit, 很大, 很慢, 接近原始[/dim]\n")

    # 量化级别对比表
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("量化级别", style="cyan")
    table.add_column("位数", justify="right")
    table.add_column("相对大小", justify="right")
    table.add_column("相对速度", justify="right")
    table.add_column("质量", justify="right")
    table.add_column("推荐场景")

    quantization_info = [
        ("Q2_K", "2-bit", "25%", "最快", "★★☆☆☆", "测试/实验"),
        ("Q3_K_M", "3-bit", "38%", "很快", "★★★☆☆", "资源受限"),
        ("Q4_K_M", "4-bit", "50%", "快", "★★★★☆", "通用推荐"),
        ("Q5_K_M", "5-bit", "63%", "中", "★★★★★", "质量优先"),
        ("Q6_K", "6-bit", "75%", "慢", "★★★★★", "高质量需求"),
        ("Q8_0", "8-bit", "100%", "最慢", "★★★★★", "接近原始"),
    ]

    for info in quantization_info:
        table.add_row(*info)

    console.print(table)


def demo_gguf_quantization():
    """演示 GGUF 量化（Ollama 默认）"""
    console.print("\n[bold cyan]🚀 演示2: GGUF 量化测试[/bold cyan]\n")

    tester = QuantizationTester()

    # 测试提示
    test_prompt = "Explain quantum computing in simple terms."
    expected_keywords = ["quantum", "computing", "bits", "superposition", "information"]

    # 要测试的量化版本
    quantizations = [
        "llama3:8b-q4_K_M",
        "llama3:8b-q5_K_M",
        "llama3:8b-q6_K"
    ]

    console.print(f"[yellow]测试提示:[/yellow] {test_prompt}\n")

    results = []

    # 获取已安装的模型
    installed_models = tester.list_models()
    installed_names = [m.get("name") for m in installed_models]

    for quant in quantizations:
        if quant not in installed_names:
            console.print(f"[yellow]⚠️  模型 {quant} 未安装，跳过[/yellow]")
            console.print(f"[dim]   安装命令: ollama pull {quant}[/dim]\n")
            continue

        console.print(f"[yellow]测试 {quant}...[/yellow]")

        result = tester.test_model(quant, test_prompt)

        if result:
            # 计算速度
            eval_count = result["eval_count"]
            eval_duration = result["eval_duration"] / 1e9  # 纳秒转秒
            speed = eval_count / eval_duration if eval_duration > 0 else 0

            # 评估质量
            quality = tester.evaluate_quality(result["response"], expected_keywords)

            # 获取模型大小
            model_info = next((m for m in installed_models if m.get("name") == quant), None)
            size_gb = model_info.get("size", 0) / (1024 ** 3) if model_info else 0

            results.append(QuantizationResult(
                model_name=quant,
                quantization=quant.split("-")[-1],
                size_gb=size_gb,
                speed_tokens_per_sec=speed,
                latency_sec=result["elapsed"],
                quality_score=quality,
                response=result["response"]
            ))

            console.print(f"[green]✅ 完成[/green]")
            console.print(f"[dim]   速度: {speed:.1f} tokens/s[/dim]")
            console.print(f"[dim]   延迟: {result['elapsed']:.2f}s[/dim]")
            console.print(f"[dim]   质量: {quality:.2f}[/dim]\n")

    # 显示对比结果
    if results:
        console.print("\n[bold]📊 GGUF 量化对比:[/bold]\n")

        comparison_table = Table(show_header=True, header_style="bold magenta")
        comparison_table.add_column("量化", style="cyan")
        comparison_table.add_column("大小 (GB)", justify="right")
        comparison_table.add_column("速度 (tok/s)", justify="right")
        comparison_table.add_column("延迟 (s)", justify="right")
        comparison_table.add_column("质量", justify="right")

        for r in results:
            comparison_table.add_row(
                r.quantization,
                f"{r.size_gb:.2f}",
                f"{r.speed_tokens_per_sec:.1f}",
                f"{r.latency_sec:.2f}",
                f"{r.quality_score:.2f}"
            )

        console.print(comparison_table)

        # 显示响应示例
        console.print("\n[bold]📝 响应示例:[/bold]\n")
        for r in results[:2]:  # 只显示前两个
            console.print(f"[yellow]{r.quantization}:[/yellow]")
            console.print(f"[green]{r.response[:200]}...[/green]\n")


def demo_quantization_formats():
    """演示不同量化格式对比"""
    console.print("\n[bold cyan]🚀 演示3: 量化格式对比[/bold cyan]\n")

    console.print("[bold]量化格式特点:[/bold]\n")

    # GGUF
    console.print("[yellow]1. GGUF (GPT-Generated Unified Format)[/yellow]")
    console.print("[dim]   优点:[/dim]")
    console.print("[dim]   - 通用格式，支持多种模型[/dim]")
    console.print("[dim]   - 支持 CPU 推理[/dim]")
    console.print("[dim]   - Ollama 默认格式[/dim]")
    console.print("[dim]   - 易于使用[/dim]")
    console.print("[dim]   缺点:[/dim]")
    console.print("[dim]   - GPU 性能不如 GPTQ/AWQ[/dim]")
    console.print("[dim]   适用: 本地部署、CPU 推理、快速原型[/dim]\n")

    # GPTQ
    console.print("[yellow]2. GPTQ (GPT Quantization)[/yellow]")
    console.print("[dim]   优点:[/dim]")
    console.print("[dim]   - GPU 推理快[/dim]")
    console.print("[dim]   - 4-bit 量化精度高[/dim]")
    console.print("[dim]   - 内存占用小[/dim]")
    console.print("[dim]   缺点:[/dim]")
    console.print("[dim]   - 只支持 GPU[/dim]")
    console.print("[dim]   - 需要 CUDA[/dim]")
    console.print("[dim]   适用: GPU 推理、生产环境[/dim]\n")

    # AWQ
    console.print("[yellow]3. AWQ (Activation-aware Weight Quantization)[/yellow]")
    console.print("[dim]   优点:[/dim]")
    console.print("[dim]   - 精度损失最小[/dim]")
    console.print("[dim]   - 激活感知量化[/dim]")
    console.print("[dim]   - 质量最好[/dim]")
    console.print("[dim]   缺点:[/dim]")
    console.print("[dim]   - 速度稍慢于 GPTQ[/dim]")
    console.print("[dim]   - 只支持 GPU[/dim]")
    console.print("[dim]   适用: 质量优先、GPU 推理[/dim]\n")

    # 对比表格
    console.print("[bold]📊 格式对比:[/bold]\n")

    format_table = Table(show_header=True, header_style="bold magenta")
    format_table.add_column("特性", style="cyan")
    format_table.add_column("GGUF")
    format_table.add_column("GPTQ")
    format_table.add_column("AWQ")

    comparisons = [
        ("CPU 支持", "✅", "❌", "❌"),
        ("GPU 支持", "✅", "✅", "✅"),
        ("推理速度", "中", "快", "中快"),
        ("量化精度", "中", "高", "最高"),
        ("内存占用", "中", "低", "低"),
        ("易用性", "高", "中", "中"),
        ("生态支持", "好", "很好", "好"),
    ]

    for comp in comparisons:
        format_table.add_row(*comp)

    console.print(format_table)


def demo_quality_evaluation():
    """演示质量评估"""
    console.print("\n[bold cyan]🚀 演示4: 量化质量评估[/bold cyan]\n")

    tester = QuantizationTester()

    # 多个测试用例
    test_cases = [
        {
            "prompt": "What is machine learning?",
            "keywords": ["machine", "learning", "data", "algorithm", "model"],
            "category": "定义类"
        },
        {
            "prompt": "Explain the difference between AI and ML.",
            "keywords": ["artificial", "intelligence", "machine", "learning", "difference"],
            "category": "对比类"
        },
        {
            "prompt": "Write a Python function to calculate factorial.",
            "keywords": ["def", "factorial", "return", "function"],
            "category": "代码类"
        }
    ]

    quantizations = ["llama3:8b-q4_K_M", "llama3:8b-q5_K_M"]

    # 获取已安装的模型
    installed_models = tester.list_models()
    installed_names = [m.get("name") for m in installed_models]

    # 过滤已安装的模型
    available_quants = [q for q in quantizations if q in installed_names]

    if not available_quants:
        console.print("[yellow]⚠️  没有可用的测试模型[/yellow]")
        return

    console.print(f"[yellow]测试 {len(test_cases)} 个用例...[/yellow]\n")

    results = {q: [] for q in available_quants}

    for i, test_case in enumerate(test_cases, 1):
        console.print(f"[bold]测试用例 {i}: {test_case['category']}[/bold]")
        console.print(f"[dim]{test_case['prompt']}[/dim]\n")

        for quant in available_quants:
            console.print(f"[yellow]  {quant}...[/yellow]", end=" ")

            result = tester.test_model(quant, test_case["prompt"])

            if result:
                quality = tester.evaluate_quality(
                    result["response"],
                    test_case["keywords"]
                )

                results[quant].append(quality)
                console.print(f"[green]✅ 质量: {quality:.2f}[/green]")
            else:
                console.print("[red]❌ 失败[/red]")

        console.print()

    # 显示平均质量
    console.print("[bold]📊 平均质量评分:[/bold]\n")

    quality_table = Table(show_header=True, header_style="bold magenta")
    quality_table.add_column("量化", style="cyan")
    quality_table.add_column("平均质量", justify="right")
    quality_table.add_column("评级")

    for quant in available_quants:
        if results[quant]:
            avg_quality = sum(results[quant]) / len(results[quant])

            # 评级
            if avg_quality >= 0.8:
                rating = "优秀 ⭐⭐⭐⭐⭐"
            elif avg_quality >= 0.6:
                rating = "良好 ⭐⭐⭐⭐"
            elif avg_quality >= 0.4:
                rating = "一般 ⭐⭐⭐"
            else:
                rating = "较差 ⭐⭐"

            quality_table.add_row(
                quant.split("-")[-1],
                f"{avg_quality:.2f}",
                rating
            )

    console.print(quality_table)


def demo_selection_guide():
    """演示量化选择指南"""
    console.print("\n[bold cyan]🚀 演示5: 量化选择指南[/bold cyan]\n")

    console.print("[bold]根据场景选择量化:[/bold]\n")

    # 场景推荐
    scenarios = [
        {
            "scenario": "快速原型开发",
            "hardware": "笔记本 (8GB RAM)",
            "recommendation": "Q4_K_M",
            "reason": "平衡速度和质量，内存占用适中"
        },
        {
            "scenario": "生产环境 (低延迟)",
            "hardware": "服务器 (GPU)",
            "recommendation": "GPTQ 4-bit",
            "reason": "GPU 推理快，延迟低"
        },
        {
            "scenario": "生产环境 (高质量)",
            "hardware": "服务器 (GPU)",
            "recommendation": "AWQ 4-bit",
            "reason": "精度损失最小，质量最好"
        },
        {
            "scenario": "资源受限设备",
            "hardware": "边缘设备 (4GB RAM)",
            "recommendation": "Q3_K_M",
            "reason": "内存占用小，仍可接受的质量"
        },
        {
            "scenario": "CPU 推理",
            "hardware": "无 GPU 服务器",
            "recommendation": "Q5_K_M (GGUF)",
            "reason": "GGUF 支持 CPU，Q5 质量好"
        },
        {
            "scenario": "实验测试",
            "hardware": "任意",
            "recommendation": "Q2_K",
            "reason": "最快，适合快速迭代"
        }
    ]

    scenario_table = Table(show_header=True, header_style="bold magenta")
    scenario_table.add_column("使用场景", style="cyan")
    scenario_table.add_column("硬件配置")
    scenario_table.add_column("推荐量化", style="green")
    scenario_table.add_column("理由")

    for s in scenarios:
        scenario_table.add_row(
            s["scenario"],
            s["hardware"],
            s["recommendation"],
            s["reason"]
        )

    console.print(scenario_table)

    # 决策树
    console.print("\n[bold]🌳 决策树:[/bold]\n")

    console.print("[yellow]1. 是否有 GPU？[/yellow]")
    console.print("   [green]是[/green] → 继续")
    console.print("   [red]否[/red] → 使用 GGUF (Q4_K_M 或 Q5_K_M)\n")

    console.print("[yellow]2. 优先考虑什么？[/yellow]")
    console.print("   [green]速度[/green] → GPTQ 4-bit")
    console.print("   [green]质量[/green] → AWQ 4-bit")
    console.print("   [green]平衡[/green] → GGUF Q5_K_M\n")

    console.print("[yellow]3. 内存是否受限？[/yellow]")
    console.print("   [green]是[/green] → 使用更激进的量化 (Q3/Q4)")
    console.print("   [red]否[/red] → 使用更高精度 (Q5/Q6)\n")


def demo_benchmark_comparison():
    """演示性能基准对比"""
    console.print("\n[bold cyan]🚀 演示6: 性能基准对比[/bold cyan]\n")

    tester = QuantizationTester()

    # 基准测试配置
    benchmark_prompts = [
        "What is AI?",
        "Explain machine learning.",
        "What is deep learning?",
        "Define neural networks.",
        "What is NLP?"
    ]

    quantizations = ["llama3:8b-q4_K_M", "llama3:8b-q5_K_M"]

    # 获取已安装的模型
    installed_models = tester.list_models()
    installed_names = [m.get("name") for m in installed_models]

    available_quants = [q for q in quantizations if q in installed_names]

    if not available_quants:
        console.print("[yellow]⚠️  没有可用的测试模型[/yellow]")
        return

    console.print(f"[yellow]运行基准测试 ({len(benchmark_prompts)} 个请求)...[/yellow]\n")

    benchmark_results = {}

    for quant in available_quants:
        console.print(f"[yellow]测试 {quant}...[/yellow]")

        speeds = []
        latencies = []

        for prompt in benchmark_prompts:
            result = tester.test_model(quant, prompt, max_tokens=50)

            if result:
                eval_count = result["eval_count"]
                eval_duration = result["eval_duration"] / 1e9
                speed = eval_count / eval_duration if eval_duration > 0 else 0

                speeds.append(speed)
                latencies.append(result["elapsed"])

        if speeds:
            benchmark_results[quant] = {
                "avg_speed": sum(speeds) / len(speeds),
                "avg_latency": sum(latencies) / len(latencies),
                "max_speed": max(speeds),
                "min_speed": min(speeds)
            }

            console.print(f"[green]✅ 完成[/green]")
            console.print(f"[dim]   平均速度: {benchmark_results[quant]['avg_speed']:.1f} tokens/s[/dim]")
            console.print(f"[dim]   平均延迟: {benchmark_results[quant]['avg_latency']:.2f}s[/dim]\n")

    # 显示对比
    if benchmark_results:
        console.print("[bold]📊 基准测试结果:[/bold]\n")

        benchmark_table = Table(show_header=True, header_style="bold magenta")
        benchmark_table.add_column("量化", style="cyan")
        benchmark_table.add_column("平均速度", justify="right")
        benchmark_table.add_column("平均延迟", justify="right")
        benchmark_table.add_column("速度范围", justify="right")

        for quant, results in benchmark_results.items():
            benchmark_table.add_row(
                quant.split("-")[-1],
                f"{results['avg_speed']:.1f} tok/s",
                f"{results['avg_latency']:.2f}s",
                f"{results['min_speed']:.1f}-{results['max_speed']:.1f}"
            )

        console.print(benchmark_table)


def main():
    """主函数"""
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]量化方法对比示例[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    demos = [
        ("量化级别说明", demo_quantization_levels),
        ("GGUF 量化测试", demo_gguf_quantization),
        ("量化格式对比", demo_quantization_formats),
        ("质量评估", demo_quality_evaluation),
        ("选择指南", demo_selection_guide),
        ("性能基准", demo_benchmark_comparison)
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
