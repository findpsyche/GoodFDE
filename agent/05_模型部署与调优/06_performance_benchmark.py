"""
性能基准测试

本示例展示如何：
1. 设计性能测试方案
2. 测试吞吐量和延迟
3. 测试并发处理能力
4. 对比不同配置
5. 生成性能报告

学习目标：
- 掌握性能测试方法
- 理解关键性能指标
- 学会分析测试结果
"""

import os
import time
import json
import statistics
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

load_dotenv()

console = Console()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    model_name: str
    test_name: str
    num_requests: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    p50_time: float
    p95_time: float
    p99_time: float
    throughput: float  # requests/s
    avg_tokens_per_sec: float
    success_rate: float
    error_count: int


class PerformanceBenchmark:
    """性能基准测试工具"""

    def __init__(self, ollama_host: str = OLLAMA_HOST):
        self.ollama_host = ollama_host
        self.api_base = f"{ollama_host}/api"

    def single_request(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 100
    ) -> Optional[Dict[str, Any]]:
        """单个请求"""
        try:
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens}
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
                eval_count = result.get("eval_count", 0)
                eval_duration = result.get("eval_duration", 0) / 1e9

                return {
                    "success": True,
                    "elapsed": elapsed,
                    "tokens": eval_count,
                    "tokens_per_sec": eval_count / eval_duration if eval_duration > 0 else 0
                }
            else:
                return {"success": False, "elapsed": elapsed}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def benchmark_throughput(
        self,
        model: str,
        prompts: List[str],
        max_tokens: int = 100
    ) -> BenchmarkResult:
        """吞吐量测试"""
        console.print(f"[yellow]🚀 测试吞吐量: {model}[/yellow]")

        times = []
        speeds = []
        errors = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("测试中...", total=len(prompts))

            start_time = time.time()

            for prompt in prompts:
                result = self.single_request(model, prompt, max_tokens)

                if result and result.get("success"):
                    times.append(result["elapsed"])
                    speeds.append(result["tokens_per_sec"])
                else:
                    errors += 1

                progress.advance(task)

            total_time = time.time() - start_time

        # 计算统计数据
        if times:
            sorted_times = sorted(times)
            n = len(sorted_times)

            return BenchmarkResult(
                model_name=model,
                test_name="throughput",
                num_requests=len(prompts),
                total_time=total_time,
                avg_time=statistics.mean(times),
                min_time=min(times),
                max_time=max(times),
                p50_time=sorted_times[int(n * 0.5)],
                p95_time=sorted_times[int(n * 0.95)],
                p99_time=sorted_times[int(n * 0.99)],
                throughput=len(prompts) / total_time,
                avg_tokens_per_sec=statistics.mean(speeds),
                success_rate=len(times) / len(prompts),
                error_count=errors
            )
        else:
            return None

    def benchmark_concurrent(
        self,
        model: str,
        prompts: List[str],
        num_workers: int = 10,
        max_tokens: int = 100
    ) -> BenchmarkResult:
        """并发测试"""
        console.print(f"[yellow]🚀 测试并发 ({num_workers} 线程): {model}[/yellow]")

        times = []
        speeds = []
        errors = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("测试中...", total=len(prompts))

            start_time = time.time()

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(self.single_request, model, prompt, max_tokens)
                    for prompt in prompts
                ]

                for future in as_completed(futures):
                    result = future.result()

                    if result and result.get("success"):
                        times.append(result["elapsed"])
                        speeds.append(result["tokens_per_sec"])
                    else:
                        errors += 1

                    progress.advance(task)

            total_time = time.time() - start_time

        # 计算统计数据
        if times:
            sorted_times = sorted(times)
            n = len(sorted_times)

            return BenchmarkResult(
                model_name=model,
                test_name=f"concurrent_{num_workers}",
                num_requests=len(prompts),
                total_time=total_time,
                avg_time=statistics.mean(times),
                min_time=min(times),
                max_time=max(times),
                p50_time=sorted_times[int(n * 0.5)],
                p95_time=sorted_times[int(n * 0.95)],
                p99_time=sorted_times[int(n * 0.99)],
                throughput=len(prompts) / total_time,
                avg_tokens_per_sec=statistics.mean(speeds),
                success_rate=len(times) / len(prompts),
                error_count=errors
            )
        else:
            return None

    def benchmark_latency(
        self,
        model: str,
        prompt: str,
        num_runs: int = 10,
        max_tokens: int = 100
    ) -> BenchmarkResult:
        """延迟测试"""
        console.print(f"[yellow]🚀 测试延迟: {model}[/yellow]")

        times = []
        speeds = []
        errors = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("测试中...", total=num_runs)

            start_time = time.time()

            for _ in range(num_runs):
                result = self.single_request(model, prompt, max_tokens)

                if result and result.get("success"):
                    times.append(result["elapsed"])
                    speeds.append(result["tokens_per_sec"])
                else:
                    errors += 1

                progress.advance(task)

            total_time = time.time() - start_time

        # 计算统计数据
        if times:
            sorted_times = sorted(times)
            n = len(sorted_times)

            return BenchmarkResult(
                model_name=model,
                test_name="latency",
                num_requests=num_runs,
                total_time=total_time,
                avg_time=statistics.mean(times),
                min_time=min(times),
                max_time=max(times),
                p50_time=sorted_times[int(n * 0.5)],
                p95_time=sorted_times[int(n * 0.95)],
                p99_time=sorted_times[int(n * 0.99)],
                throughput=num_runs / total_time,
                avg_tokens_per_sec=statistics.mean(speeds),
                success_rate=len(times) / num_runs,
                error_count=errors
            )
        else:
            return None


def demo_throughput_test():
    """演示吞吐量测试"""
    console.print("\n[bold cyan]🚀 演示1: 吞吐量测试[/bold cyan]\n")

    benchmark = PerformanceBenchmark()

    # 测试数据
    test_prompts = [
        "What is artificial intelligence?",
        "Explain machine learning.",
        "What is deep learning?",
        "Define neural networks.",
        "What is natural language processing?",
        "Explain computer vision.",
        "What is reinforcement learning?",
        "Define supervised learning.",
        "What is unsupervised learning?",
        "Explain transfer learning."
    ]

    model = "llama3:8b"

    result = benchmark.benchmark_throughput(model, test_prompts, max_tokens=50)

    if result:
        console.print("\n[bold]📊 吞吐量测试结果:[/bold]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan")
        table.add_column("数值", justify="right", style="green")

        table.add_row("总请求数", f"{result.num_requests}")
        table.add_row("总耗时", f"{result.total_time:.2f}s")
        table.add_row("吞吐量", f"{result.throughput:.2f} req/s")
        table.add_row("平均延迟", f"{result.avg_time:.2f}s")
        table.add_row("P50 延迟", f"{result.p50_time:.2f}s")
        table.add_row("P95 延迟", f"{result.p95_time:.2f}s")
        table.add_row("P99 延迟", f"{result.p99_time:.2f}s")
        table.add_row("平均速度", f"{result.avg_tokens_per_sec:.1f} tok/s")
        table.add_row("成功率", f"{result.success_rate * 100:.1f}%")

        console.print(table)


def demo_concurrent_test():
    """演示并发测试"""
    console.print("\n[bold cyan]🚀 演示2: 并发测试[/bold cyan]\n")

    benchmark = PerformanceBenchmark()

    test_prompts = [f"Question {i}: Explain AI topic {i}." for i in range(20)]

    model = "llama3:8b"

    # 测试不同并发级别
    concurrent_levels = [1, 5, 10]
    results = []

    for num_workers in concurrent_levels:
        result = benchmark.benchmark_concurrent(
            model,
            test_prompts,
            num_workers=num_workers,
            max_tokens=50
        )

        if result:
            results.append(result)

    # 显示对比
    if results:
        console.print("\n[bold]📊 并发测试对比:[/bold]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("并发数", justify="right")
        table.add_column("吞吐量", justify="right")
        table.add_column("平均延迟", justify="right")
        table.add_column("P95 延迟", justify="right")
        table.add_column("成功率", justify="right")

        for r in results:
            workers = r.test_name.split("_")[-1]
            table.add_row(
                workers,
                f"{r.throughput:.2f} req/s",
                f"{r.avg_time:.2f}s",
                f"{r.p95_time:.2f}s",
                f"{r.success_rate * 100:.1f}%"
            )

        console.print(table)


def demo_latency_test():
    """演示延迟测试"""
    console.print("\n[bold cyan]🚀 演示3: 延迟测试[/bold cyan]\n")

    benchmark = PerformanceBenchmark()

    prompt = "Explain quantum computing in simple terms."
    model = "llama3:8b"

    result = benchmark.benchmark_latency(model, prompt, num_runs=10, max_tokens=100)

    if result:
        console.print("\n[bold]📊 延迟测试结果:[/bold]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan")
        table.add_column("数值", justify="right", style="green")

        table.add_row("测试次数", f"{result.num_requests}")
        table.add_row("平均延迟", f"{result.avg_time:.3f}s")
        table.add_row("最小延迟", f"{result.min_time:.3f}s")
        table.add_row("最大延迟", f"{result.max_time:.3f}s")
        table.add_row("P50 延迟", f"{result.p50_time:.3f}s")
        table.add_row("P95 延迟", f"{result.p95_time:.3f}s")
        table.add_row("P99 延迟", f"{result.p99_time:.3f}s")
        table.add_row("标准差", f"{statistics.stdev([result.min_time, result.max_time]):.3f}s")

        console.print(table)


def demo_model_comparison():
    """演示模型对比"""
    console.print("\n[bold cyan]🚀 演示4: 模型对比[/bold cyan]\n")

    benchmark = PerformanceBenchmark()

    # 要对比的模型
    models = ["llama3:8b-q4_K_M", "llama3:8b-q5_K_M"]

    test_prompts = [
        "What is AI?",
        "Explain ML.",
        "What is DL?",
        "Define NN.",
        "What is NLP?"
    ]

    results = []

    for model in models:
        console.print(f"\n[yellow]测试 {model}...[/yellow]")
        result = benchmark.benchmark_throughput(model, test_prompts, max_tokens=50)

        if result:
            results.append(result)

    # 显示对比
    if results:
        console.print("\n[bold]📊 模型性能对比:[/bold]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("模型", style="cyan")
        table.add_column("吞吐量", justify="right")
        table.add_column("平均延迟", justify="right")
        table.add_column("速度", justify="right")

        for r in results:
            table.add_row(
                r.model_name.split("-")[-1],
                f"{r.throughput:.2f} req/s",
                f"{r.avg_time:.2f}s",
                f"{r.avg_tokens_per_sec:.1f} tok/s"
            )

        console.print(table)


def demo_report_generation():
    """演示报告生成"""
    console.print("\n[bold cyan]🚀 演示5: 生成性能报告[/bold cyan]\n")

    # 模拟测试结果
    results = [
        BenchmarkResult(
            model_name="llama3:8b-q4_K_M",
            test_name="throughput",
            num_requests=10,
            total_time=25.5,
            avg_time=2.55,
            min_time=2.1,
            max_time=3.2,
            p50_time=2.5,
            p95_time=3.0,
            p99_time=3.1,
            throughput=0.39,
            avg_tokens_per_sec=35.2,
            success_rate=1.0,
            error_count=0
        )
    ]

    # 生成报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_environment": {
            "ollama_host": OLLAMA_HOST,
            "platform": "local"
        },
        "results": [asdict(r) for r in results]
    }

    # 保存报告
    report_file = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✅ 报告已保存: {report_file}[/green]")

    # 显示报告摘要
    console.print("\n[bold]📄 报告摘要:[/bold]\n")
    console.print(json.dumps(report, indent=2, ensure_ascii=False))


def demo_best_practices():
    """演示测试最佳实践"""
    console.print("\n[bold cyan]🚀 演示6: 测试最佳实践[/bold cyan]\n")

    console.print("[yellow]1. 测试设计原则:[/yellow]\n")

    principles = [
        "✅ 使用真实的工作负载",
        "✅ 测试多种场景（短/长文本）",
        "✅ 包含边界情况",
        "✅ 多次运行取平均值",
        "✅ 记录测试环境",
        "✅ 对比基准数据"
    ]

    for p in principles:
        console.print(p)

    console.print("\n[yellow]2. 关键指标:[/yellow]\n")

    metrics_table = Table(show_header=True, header_style="bold magenta")
    metrics_table.add_column("指标", style="cyan")
    metrics_table.add_column("说明")
    metrics_table.add_column("目标值")

    metrics_table.add_row(
        "吞吐量",
        "每秒处理的请求数",
        "> 10 req/s"
    )
    metrics_table.add_row(
        "P50 延迟",
        "50% 请求的响应时间",
        "< 2s"
    )
    metrics_table.add_row(
        "P95 延迟",
        "95% 请求的响应时间",
        "< 5s"
    )
    metrics_table.add_row(
        "P99 延迟",
        "99% 请求的响应时间",
        "< 10s"
    )
    metrics_table.add_row(
        "成功率",
        "成功请求的比例",
        "> 99%"
    )
    metrics_table.add_row(
        "Tokens/s",
        "生成速度",
        "> 30 tok/s"
    )

    console.print(metrics_table)

    console.print("\n[yellow]3. 测试流程:[/yellow]\n")

    workflow = [
        "1. 建立基准 → 记录当前性能",
        "2. 应用优化 → 修改配置",
        "3. 重新测试 → 对比结果",
        "4. 分析差异 → 找出原因",
        "5. 迭代优化 → 继续改进"
    ]

    for step in workflow:
        console.print(f"[green]{step}[/green]")

    console.print("\n[yellow]4. 常见陷阱:[/yellow]\n")

    pitfalls = [
        ("❌ 冷启动", "第一次请求慢，应预热"),
        ("❌ 缓存影响", "重复请求可能被缓存"),
        ("❌ 网络波动", "本地测试避免网络影响"),
        ("❌ 资源竞争", "测试时关闭其他程序"),
        ("❌ 样本太少", "至少测试 10 次以上")
    ]

    for pitfall, desc in pitfalls:
        console.print(f"{pitfall}: {desc}")


def main():
    """主函数"""
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]性能基准测试[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    demos = [
        ("吞吐量测试", demo_throughput_test),
        ("并发测试", demo_concurrent_test),
        ("延迟测试", demo_latency_test),
        ("模型对比", demo_model_comparison),
        ("生成报告", demo_report_generation),
        ("最佳实践", demo_best_practices)
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
