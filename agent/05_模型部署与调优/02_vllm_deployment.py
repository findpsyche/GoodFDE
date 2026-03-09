"""
vLLM 部署示例

本示例展示如何：
1. 使用 vLLM 部署模型
2. 配置 PagedAttention 和连续批处理
3. 对比 vLLM vs Ollama 性能
4. 测试并发处理能力
5. 优化推理参数

学习目标：
- 理解 vLLM 的核心优势
- 掌握 vLLM 的配置和优化
- 学会性能测试和对比

注意：vLLM 需要 NVIDIA GPU 支持
"""

import os
import time
import asyncio
import statistics
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# 加载环境变量
load_dotenv()

console = Console()

# 检查是否安装了 vLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    console.print("[yellow]⚠️  vLLM 未安装。请运行: pip install vllm[/yellow]")


class VLLMDeployment:
    """vLLM 部署封装"""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9
    ):
        """
        初始化 vLLM 部署

        Args:
            model_name: 模型名称或路径
            tensor_parallel_size: 张量并行大小（多GPU）
            max_model_len: 最大序列长度
            gpu_memory_utilization: GPU 内存利用率 (0-1)
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM 未安装")

        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

        console.print(f"[yellow]🚀 初始化 vLLM...[/yellow]")
        console.print(f"[dim]模型: {model_name}[/dim]")
        console.print(f"[dim]张量并行: {tensor_parallel_size}[/dim]")
        console.print(f"[dim]最大长度: {max_model_len or '自动'}[/dim]")
        console.print(f"[dim]GPU 内存利用率: {gpu_memory_utilization}[/dim]")

        # 初始化 LLM
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True
        )

        console.print("[green]✅ vLLM 初始化完成[/green]")

    def generate(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        **kwargs
    ) -> List[str]:
        """
        生成响应

        Args:
            prompts: 输入提示列表
            temperature: 温度参数
            top_p: Top-p 采样
            max_tokens: 最大生成 token 数
            **kwargs: 其他采样参数

        Returns:
            生成的文本列表
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )

        outputs = self.llm.generate(prompts, sampling_params)

        return [output.outputs[0].text for output in outputs]

    def benchmark(
        self,
        prompts: List[str],
        **sampling_kwargs
    ) -> Dict[str, Any]:
        """
        性能基准测试

        Args:
            prompts: 测试提示列表
            **sampling_kwargs: 采样参数

        Returns:
            性能统计数据
        """
        start_time = time.time()

        outputs = self.generate(prompts, **sampling_kwargs)

        elapsed = time.time() - start_time

        # 计算统计数据
        total_tokens = sum(len(output.split()) for output in outputs)
        throughput = len(prompts) / elapsed  # requests/s
        tokens_per_second = total_tokens / elapsed

        return {
            "num_requests": len(prompts),
            "total_time": elapsed,
            "throughput": throughput,
            "tokens_per_second": tokens_per_second,
            "avg_time_per_request": elapsed / len(prompts),
            "total_tokens": total_tokens
        }


def demo_basic_usage():
    """演示基本使用"""
    console.print("\n[bold cyan]🚀 演示1: vLLM 基本使用[/bold cyan]\n")

    if not VLLM_AVAILABLE:
        console.print("[red]❌ vLLM 未安装，跳过演示[/red]")
        return

    # 使用较小的模型进行演示
    model_name = "facebook/opt-125m"  # 小模型，快速测试

    console.print(f"[yellow]加载模型: {model_name}[/yellow]")

    try:
        vllm = VLLMDeployment(
            model_name=model_name,
            max_model_len=512
        )

        # 单个请求
        console.print("\n[yellow]1. 单个请求测试[/yellow]")
        prompts = ["What is artificial intelligence?"]

        start_time = time.time()
        outputs = vllm.generate(prompts, max_tokens=100)
        elapsed = time.time() - start_time

        console.print(f"[cyan]问题:[/cyan] {prompts[0]}")
        console.print(f"[cyan]回答:[/cyan] {outputs[0]}")
        console.print(f"[dim]⏱️  耗时: {elapsed:.2f}s[/dim]")

        # 批量请求
        console.print("\n[yellow]2. 批量请求测试[/yellow]")
        prompts = [
            "What is machine learning?",
            "Explain neural networks.",
            "What is deep learning?",
            "Define NLP.",
            "What is computer vision?"
        ]

        start_time = time.time()
        outputs = vllm.generate(prompts, max_tokens=100)
        elapsed = time.time() - start_time

        console.print(f"[green]✅ 处理 {len(prompts)} 个请求[/green]")
        console.print(f"[dim]⏱️  总耗时: {elapsed:.2f}s[/dim]")
        console.print(f"[dim]⏱️  平均耗时: {elapsed/len(prompts):.2f}s/请求[/dim]")
        console.print(f"[dim]🚀 吞吐量: {len(prompts)/elapsed:.2f} 请求/秒[/dim]")

    except Exception as e:
        console.print(f"[red]❌ 错误: {e}[/red]")


def demo_batch_processing():
    """演示批处理优势"""
    console.print("\n[bold cyan]🚀 演示2: 批处理性能对比[/bold cyan]\n")

    if not VLLM_AVAILABLE:
        console.print("[red]❌ vLLM 未安装，跳过演示[/red]")
        return

    model_name = "facebook/opt-125m"

    try:
        vllm = VLLMDeployment(model_name=model_name, max_model_len=512)

        # 准备测试数据
        test_prompts = [
            f"Question {i}: What is AI topic {i}?"
            for i in range(20)
        ]

        # 测试不同的批大小
        batch_sizes = [1, 4, 8, 16, 20]
        results = []

        console.print("[yellow]测试不同批大小的性能...[/yellow]\n")

        for batch_size in batch_sizes:
            console.print(f"[dim]测试 batch_size={batch_size}...[/dim]", end=" ")

            # 分批处理
            start_time = time.time()
            all_outputs = []

            for i in range(0, len(test_prompts), batch_size):
                batch = test_prompts[i:i + batch_size]
                outputs = vllm.generate(batch, max_tokens=50)
                all_outputs.extend(outputs)

            elapsed = time.time() - start_time

            throughput = len(test_prompts) / elapsed
            avg_latency = elapsed / len(test_prompts)

            results.append({
                "batch_size": batch_size,
                "total_time": elapsed,
                "throughput": throughput,
                "avg_latency": avg_latency
            })

            console.print(f"[green]✅ {throughput:.2f} req/s[/green]")

        # 显示对比表格
        console.print("\n[bold]📊 批处理性能对比:[/bold]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Batch Size", justify="right")
        table.add_column("总耗时 (s)", justify="right")
        table.add_column("吞吐量 (req/s)", justify="right")
        table.add_column("平均延迟 (s)", justify="right")
        table.add_column("提升", justify="right")

        baseline_throughput = results[0]["throughput"]

        for r in results:
            improvement = r["throughput"] / baseline_throughput

            table.add_row(
                str(r["batch_size"]),
                f"{r['total_time']:.2f}",
                f"{r['throughput']:.2f}",
                f"{r['avg_latency']:.3f}",
                f"{improvement:.2f}x"
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]❌ 错误: {e}[/red]")


def demo_concurrent_requests():
    """演示并发请求处理"""
    console.print("\n[bold cyan]🚀 演示3: 并发请求处理[/bold cyan]\n")

    if not VLLM_AVAILABLE:
        console.print("[red]❌ vLLM 未安装，跳过演示[/red]")
        return

    console.print("[yellow]模拟并发请求场景...[/yellow]")
    console.print("[dim]说明: vLLM 的连续批处理可以高效处理并发请求[/dim]\n")

    model_name = "facebook/opt-125m"

    try:
        vllm = VLLMDeployment(model_name=model_name, max_model_len=512)

        # 模拟不同并发级别
        concurrent_levels = [1, 5, 10, 20]
        results = []

        for num_concurrent in concurrent_levels:
            console.print(f"[dim]测试 {num_concurrent} 个并发请求...[/dim]", end=" ")

            prompts = [
                f"Concurrent request {i}: Explain topic {i}."
                for i in range(num_concurrent)
            ]

            start_time = time.time()
            outputs = vllm.generate(prompts, max_tokens=50)
            elapsed = time.time() - start_time

            throughput = num_concurrent / elapsed
            avg_latency = elapsed / num_concurrent

            results.append({
                "concurrent": num_concurrent,
                "total_time": elapsed,
                "throughput": throughput,
                "avg_latency": avg_latency
            })

            console.print(f"[green]✅ {throughput:.2f} req/s[/green]")

        # 显示结果
        console.print("\n[bold]📊 并发性能:[/bold]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("并发数", justify="right")
        table.add_column("总耗时 (s)", justify="right")
        table.add_column("吞吐量 (req/s)", justify="right")
        table.add_column("平均延迟 (s)", justify="right")

        for r in results:
            table.add_row(
                str(r["concurrent"]),
                f"{r['total_time']:.2f}",
                f"{r['throughput']:.2f}",
                f"{r['avg_latency']:.3f}"
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]❌ 错误: {e}[/red]")


def demo_parameter_tuning():
    """演示参数调优"""
    console.print("\n[bold cyan]🚀 演示4: 参数调优[/bold cyan]\n")

    if not VLLM_AVAILABLE:
        console.print("[red]❌ vLLM 未安装，跳过演示[/red]")
        return

    model_name = "facebook/opt-125m"

    try:
        console.print("[yellow]测试不同的采样参数...[/yellow]\n")

        vllm = VLLMDeployment(model_name=model_name, max_model_len=512)

        prompt = "Write a creative story about a robot."

        # 测试不同温度
        temperatures = [0.1, 0.7, 1.0, 1.5]

        console.print("[bold]1. 温度参数对比:[/bold]\n")

        for temp in temperatures:
            console.print(f"[yellow]Temperature = {temp}[/yellow]")

            outputs = vllm.generate(
                [prompt],
                temperature=temp,
                max_tokens=100
            )

            console.print(f"[green]{outputs[0][:150]}...[/green]\n")

        # 测试不同 top_p
        top_ps = [0.5, 0.9, 0.95, 1.0]

        console.print("\n[bold]2. Top-p 参数对比:[/bold]\n")

        for top_p in top_ps:
            console.print(f"[yellow]Top-p = {top_p}[/yellow]")

            outputs = vllm.generate(
                [prompt],
                temperature=0.7,
                top_p=top_p,
                max_tokens=100
            )

            console.print(f"[green]{outputs[0][:150]}...[/green]\n")

    except Exception as e:
        console.print(f"[red]❌ 错误: {e}[/red]")


def demo_memory_optimization():
    """演示内存优化"""
    console.print("\n[bold cyan]🚀 演示5: 内存优化[/bold cyan]\n")

    if not VLLM_AVAILABLE:
        console.print("[red]❌ vLLM 未安装，跳过演示[/red]")
        return

    console.print("[yellow]测试不同的内存配置...[/yellow]\n")

    model_name = "facebook/opt-125m"

    # 测试不同的 GPU 内存利用率
    gpu_memory_configs = [0.5, 0.7, 0.9]

    for gpu_mem in gpu_memory_configs:
        console.print(f"[yellow]GPU 内存利用率: {gpu_mem}[/yellow]")

        try:
            vllm = VLLMDeployment(
                model_name=model_name,
                max_model_len=512,
                gpu_memory_utilization=gpu_mem
            )

            # 测试性能
            prompts = [f"Test prompt {i}" for i in range(10)]

            start_time = time.time()
            outputs = vllm.generate(prompts, max_tokens=50)
            elapsed = time.time() - start_time

            console.print(f"[green]✅ 成功处理 {len(prompts)} 个请求[/green]")
            console.print(f"[dim]⏱️  耗时: {elapsed:.2f}s[/dim]")
            console.print(f"[dim]🚀 吞吐量: {len(prompts)/elapsed:.2f} req/s[/dim]\n")

        except Exception as e:
            console.print(f"[red]❌ 配置失败: {e}[/red]\n")


def compare_with_ollama():
    """对比 vLLM 和 Ollama"""
    console.print("\n[bold cyan]🚀 演示6: vLLM vs Ollama 性能对比[/bold cyan]\n")

    console.print("[yellow]说明:[/yellow]")
    console.print("[dim]- vLLM: 高性能推理引擎，优化了批处理和内存管理[/dim]")
    console.print("[dim]- Ollama: 易用的本地部署工具，适合快速原型[/dim]")
    console.print("[dim]- 预期: vLLM 在批处理和并发场景下性能更优[/dim]\n")

    # 这里只是展示对比框架，实际对比需要两个服务都运行
    console.print("[bold]性能对比维度:[/bold]")

    comparison_table = Table(show_header=True, header_style="bold magenta")
    comparison_table.add_column("维度")
    comparison_table.add_column("vLLM")
    comparison_table.add_column("Ollama")

    comparison_table.add_row(
        "单请求延迟",
        "中等",
        "低"
    )
    comparison_table.add_row(
        "批处理吞吐量",
        "高 (连续批处理)",
        "中"
    )
    comparison_table.add_row(
        "并发处理",
        "优秀 (PagedAttention)",
        "良好"
    )
    comparison_table.add_row(
        "内存效率",
        "高 (PagedAttention)",
        "中"
    )
    comparison_table.add_row(
        "易用性",
        "中 (需要配置)",
        "高 (开箱即用)"
    )
    comparison_table.add_row(
        "适用场景",
        "生产环境、高并发",
        "开发测试、快速原型"
    )

    console.print(comparison_table)

    console.print("\n[bold]建议:[/bold]")
    console.print("- 开发阶段: 使用 Ollama（快速迭代）")
    console.print("- 生产环境: 使用 vLLM（高性能）")
    console.print("- 低并发: 两者差异不大")
    console.print("- 高并发: vLLM 优势明显")


def main():
    """主函数"""
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]vLLM 部署示例[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    if not VLLM_AVAILABLE:
        console.print("\n[red]❌ vLLM 未安装[/red]")
        console.print("\n[yellow]安装方法:[/yellow]")
        console.print("  pip install vllm")
        console.print("\n[yellow]注意:[/yellow]")
        console.print("  - vLLM 需要 NVIDIA GPU")
        console.print("  - 需要 CUDA 11.8+")
        console.print("  - 推荐使用 A100, RTX 3090 或更高")
        return

    demos = [
        ("基本使用", demo_basic_usage),
        ("批处理性能", demo_batch_processing),
        ("并发请求", demo_concurrent_requests),
        ("参数调优", demo_parameter_tuning),
        ("内存优化", demo_memory_optimization),
        ("vLLM vs Ollama", compare_with_ollama)
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
