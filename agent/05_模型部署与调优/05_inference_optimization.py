"""
推理优化示例

本示例展示如何：
1. 批处理优化
2. KV Cache 优化
3. Flash Attention 加速
4. 并发处理优化
5. 内存管理优化

学习目标：
- 理解推理优化的核心技术
- 掌握性能调优方法
- 学会测量优化效果
"""

import os
import time
import asyncio
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

load_dotenv()

console = Console()


def demo_batch_processing():
    """演示批处理优化"""
    console.print("\n[bold cyan]🚀 演示1: 批处理优化[/bold cyan]\n")

    console.print("[yellow]批处理原理:[/yellow]\n")
    console.print("[dim]单个处理:[/dim]")
    console.print("  Request 1 → Process → Response 1")
    console.print("  Request 2 → Process → Response 2")
    console.print("  Request 3 → Process → Response 3")
    console.print("  总时间 = 3 × 单次时间\n")

    console.print("[dim]批处理:[/dim]")
    console.print("  [Request 1, Request 2, Request 3] → Process → [Response 1, 2, 3]")
    console.print("  总时间 ≈ 1.5 × 单次时间\n")

    console.print("[yellow]批处理权衡:[/yellow]\n")

    tradeoff_table = Table(show_header=True, header_style="bold magenta")
    tradeoff_table.add_column("Batch Size", justify="right")
    tradeoff_table.add_column("吞吐量", justify="right")
    tradeoff_table.add_column("延迟", justify="right")
    tradeoff_table.add_column("内存", justify="right")
    tradeoff_table.add_column("适用场景")

    tradeoff_table.add_row("1", "低", "最低", "最小", "实时交互")
    tradeoff_table.add_row("4-8", "中", "低", "中", "平衡选择")
    tradeoff_table.add_row("16-32", "高", "中", "大", "批量处理")
    tradeoff_table.add_row("64+", "最高", "高", "很大", "离线任务")

    console.print(tradeoff_table)

    console.print("\n[yellow]优化建议:[/yellow]\n")
    console.print("1. 实时应用: batch_size=1-4")
    console.print("2. API 服务: batch_size=4-16")
    console.print("3. 批量任务: batch_size=16-64")
    console.print("4. 使用动态批处理（vLLM）")


def demo_kv_cache():
    """演示 KV Cache 优化"""
    console.print("\n[bold cyan]🚀 演示2: KV Cache 优化[/bold cyan]\n")

    console.print("[yellow]KV Cache 原理:[/yellow]\n")
    console.print("[dim]无缓存:[/dim]")
    console.print("  Token 1: 计算 K1, V1 → 生成")
    console.print("  Token 2: 重新计算 K1, V1, K2, V2 → 生成")
    console.print("  Token 3: 重新计算 K1, V1, K2, V2, K3, V3 → 生成")
    console.print("  计算量: O(n²)\n")

    console.print("[dim]有缓存:[/dim]")
    console.print("  Token 1: 计算 K1, V1 → 缓存 → 生成")
    console.print("  Token 2: 读取 K1, V1, 计算 K2, V2 → 缓存 → 生成")
    console.print("  Token 3: 读取 K1, V1, K2, V2, 计算 K3, V3 → 缓存 → 生成")
    console.print("  计算量: O(n)\n")

    console.print("[yellow]内存占用计算:[/yellow]\n")
    console.print("KV Cache 大小 = batch_size × seq_len × num_layers × hidden_dim × 2 × dtype_size\n")

    console.print("[dim]示例: Llama 3 8B[/dim]")
    console.print("  num_layers = 32")
    console.print("  hidden_dim = 4096")
    console.print("  dtype = FP16 (2 bytes)\n")

    memory_table = Table(show_header=True, header_style="bold magenta")
    memory_table.add_column("Batch Size", justify="right")
    memory_table.add_column("Seq Len", justify="right")
    memory_table.add_column("KV Cache (GB)", justify="right")

    configs = [
        (1, 2048, 1.0),
        (1, 4096, 2.0),
        (4, 2048, 4.0),
        (8, 2048, 8.0),
        (16, 2048, 16.0)
    ]

    for batch, seq_len, memory in configs:
        memory_table.add_row(str(batch), str(seq_len), f"{memory:.1f}")

    console.print(memory_table)

    console.print("\n[yellow]优化策略:[/yellow]\n")
    console.print("1. 限制 max_model_len")
    console.print("2. 使用滑动窗口")
    console.print("3. PagedAttention (vLLM)")
    console.print("4. 定期清理过期缓存")


def demo_flash_attention():
    """演示 Flash Attention"""
    console.print("\n[bold cyan]🚀 演示3: Flash Attention[/bold cyan]\n")

    console.print("[yellow]Flash Attention 原理:[/yellow]\n")
    console.print("[dim]传统 Attention:[/dim]")
    console.print("  1. 计算 Q @ K^T → 写入 HBM")
    console.print("  2. Softmax → 读取 HBM → 写入 HBM")
    console.print("  3. @ V → 读取 HBM → 写入 HBM")
    console.print("  HBM 访问次数: 多次\n")

    console.print("[dim]Flash Attention:[/dim]")
    console.print("  1. 分块计算，保持在 SRAM")
    console.print("  2. 减少 HBM 访问")
    console.print("  3. 重计算代替存储")
    console.print("  HBM 访问次数: 最少\n")

    console.print("[yellow]性能提升:[/yellow]\n")

    performance_table = Table(show_header=True, header_style="bold magenta")
    performance_table.add_column("指标")
    performance_table.add_column("传统 Attention")
    performance_table.add_column("Flash Attention")
    performance_table.add_column("提升")

    performance_table.add_row("速度", "1x", "2-4x", "2-4x")
    performance_table.add_row("内存", "O(n²)", "O(n)", "大幅降低")
    performance_table.add_row("精度", "FP16", "FP16", "相同")

    console.print(performance_table)

    console.print("\n[yellow]硬件要求:[/yellow]\n")
    console.print("✅ NVIDIA Ampere 架构以上")
    console.print("✅ A100, A6000, RTX 3090, RTX 4090")
    console.print("❌ V100, RTX 2080 (不支持)\n")

    console.print("[yellow]使用方法:[/yellow]\n")
    console.print("```bash")
    console.print("# 安装")
    console.print("pip install flash-attn")
    console.print("")
    console.print("# vLLM 自动启用")
    console.print("vllm serve model_name")
    console.print("")
    console.print("# 检查是否启用")
    console.print("# 查看日志中的 'Using Flash Attention'")
    console.print("```")


def demo_concurrent_optimization():
    """演示并发优化"""
    console.print("\n[bold cyan]🚀 演示4: 并发处理优化[/bold cyan]\n")

    console.print("[yellow]并发模式对比:[/yellow]\n")

    console.print("[dim]1. 串行处理:[/dim]")
    console.print("  Request 1 → Process → Response 1")
    console.print("  Request 2 → Process → Response 2")
    console.print("  总时间 = T1 + T2\n")

    console.print("[dim]2. 多线程:[/dim]")
    console.print("  Thread 1: Request 1 → Process → Response 1")
    console.print("  Thread 2: Request 2 → Process → Response 2")
    console.print("  总时间 = max(T1, T2)\n")

    console.print("[dim]3. 异步处理:[/dim]")
    console.print("  async def process(request):")
    console.print("      await model.generate(request)")
    console.print("  await asyncio.gather(*tasks)\n")

    console.print("[yellow]并发策略:[/yellow]\n")

    strategy_table = Table(show_header=True, header_style="bold magenta")
    strategy_table.add_column("策略", style="cyan")
    strategy_table.add_column("优点")
    strategy_table.add_column("缺点")
    strategy_table.add_column("适用场景")

    strategy_table.add_row(
        "串行",
        "简单、稳定",
        "吞吐量低",
        "低并发"
    )
    strategy_table.add_row(
        "多线程",
        "易实现",
        "GIL 限制",
        "I/O 密集"
    )
    strategy_table.add_row(
        "多进程",
        "真并行",
        "内存占用大",
        "CPU 密集"
    )
    strategy_table.add_row(
        "异步",
        "高效、轻量",
        "复杂度高",
        "高并发"
    )
    strategy_table.add_row(
        "连续批处理",
        "最优",
        "需要支持",
        "vLLM"
    )

    console.print(strategy_table)


def demo_memory_optimization():
    """演示内存优化"""
    console.print("\n[bold cyan]🚀 演示5: 内存管理优化[/bold cyan]\n")

    console.print("[yellow]内存占用分析:[/yellow]\n")

    memory_breakdown = [
        ("模型权重", "4-40 GB", "取决于模型大小和量化"),
        ("KV Cache", "1-20 GB", "取决于 batch size 和序列长度"),
        ("激活值", "0.5-2 GB", "取决于 batch size"),
        ("梯度", "0 GB", "推理时不需要"),
        ("优化器状态", "0 GB", "推理时不需要")
    ]

    memory_table = Table(show_header=True, header_style="bold magenta")
    memory_table.add_column("组件", style="cyan")
    memory_table.add_column("占用", justify="right")
    memory_table.add_column("说明")

    for component, size, desc in memory_breakdown:
        memory_table.add_row(component, size, desc)

    console.print(memory_table)

    console.print("\n[yellow]优化策略:[/yellow]\n")

    optimizations = [
        ("1. 模型量化", "4-bit/8-bit 量化", "减少 50-75% 内存"),
        ("2. 限制序列长度", "max_model_len=2048", "减少 KV Cache"),
        ("3. 减小 batch size", "batch_size=4", "减少激活值内存"),
        ("4. PagedAttention", "vLLM", "高效 KV Cache 管理"),
        ("5. 卸载到 CPU", "offload", "牺牲速度换内存"),
        ("6. 模型分片", "tensor_parallel", "多 GPU 分担")
    ]

    for name, method, benefit in optimizations:
        console.print(f"[green]{name}:[/green] {method}")
        console.print(f"  [dim]效果: {benefit}[/dim]\n")


def demo_optimization_workflow():
    """演示优化工作流"""
    console.print("\n[bold cyan]🚀 演示6: 优化工作流[/bold cyan]\n")

    console.print("[yellow]优化步骤:[/yellow]\n")

    workflow = [
        {
            "step": "1. 建立基准",
            "actions": [
                "测试默认配置的性能",
                "记录吞吐量、延迟、内存",
                "确定优化目标"
            ]
        },
        {
            "step": "2. 识别瓶颈",
            "actions": [
                "使用 profiler 分析",
                "检查 GPU 利用率",
                "检查内存使用",
                "找到最慢的部分"
            ]
        },
        {
            "step": "3. 应用优化",
            "actions": [
                "一次优化一个方面",
                "测量每次优化的效果",
                "记录配置变化"
            ]
        },
        {
            "step": "4. 验证效果",
            "actions": [
                "重新测试性能",
                "对比优化前后",
                "检查质量是否下降"
            ]
        },
        {
            "step": "5. 迭代优化",
            "actions": [
                "继续优化其他瓶颈",
                "平衡速度、质量、成本",
                "达到目标后停止"
            ]
        }
    ]

    for item in workflow:
        console.print(f"[bold]{item['step']}[/bold]")
        for action in item['actions']:
            console.print(f"  • {action}")
        console.print()

    console.print("[yellow]优化检查清单:[/yellow]\n")

    checklist = [
        "[ ] 启用 Flash Attention",
        "[ ] 使用合适的 batch size",
        "[ ] 限制 max_model_len",
        "[ ] 使用模型量化",
        "[ ] 配置 GPU 内存利用率",
        "[ ] 使用连续批处理（vLLM）",
        "[ ] 启用 KV Cache",
        "[ ] 优化并发处理",
        "[ ] 监控资源使用",
        "[ ] 测试边界情况"
    ]

    for item in checklist:
        console.print(item)


def demo_performance_comparison():
    """演示性能对比"""
    console.print("\n[bold cyan]🚀 演示7: 性能对比示例[/bold cyan]\n")

    console.print("[yellow]优化前后对比:[/yellow]\n")

    comparison_table = Table(show_header=True, header_style="bold magenta")
    comparison_table.add_column("配置", style="cyan")
    comparison_table.add_column("吞吐量", justify="right")
    comparison_table.add_column("延迟", justify="right")
    comparison_table.add_column("内存", justify="right")

    configs = [
        ("基准 (Ollama)", "10 req/s", "2.0s", "8 GB"),
        ("+ 批处理 (batch=8)", "25 req/s", "2.5s", "12 GB"),
        ("+ vLLM", "40 req/s", "1.5s", "10 GB"),
        ("+ Flash Attention", "60 req/s", "1.0s", "8 GB"),
        ("+ 量化 (Q4)", "70 req/s", "0.8s", "5 GB")
    ]

    for config, throughput, latency, memory in configs:
        comparison_table.add_row(config, throughput, latency, memory)

    console.print(comparison_table)

    console.print("\n[yellow]提升总结:[/yellow]\n")
    console.print("• 吞吐量: 10 → 70 req/s (7x)")
    console.print("• 延迟: 2.0s → 0.8s (2.5x)")
    console.print("• 内存: 8 GB → 5 GB (节省 37.5%)")


def demo_best_practices():
    """演示最佳实践"""
    console.print("\n[bold cyan]🚀 演示8: 最佳实践[/bold cyan]\n")

    console.print("[yellow]1. 开发阶段:[/yellow]\n")
    console.print("• 使用小模型快速迭代")
    console.print("• 使用 Ollama 快速原型")
    console.print("• 不过度优化\n")

    console.print("[yellow]2. 测试阶段:[/yellow]\n")
    console.print("• 建立性能基准")
    console.print("• 测试不同配置")
    console.print("• 记录优化效果\n")

    console.print("[yellow]3. 生产阶段:[/yellow]\n")
    console.print("• 使用 vLLM 或 TGI")
    console.print("• 启用所有优化")
    console.print("• 持续监控性能\n")

    console.print("[yellow]4. 常见错误:[/yellow]\n")

    mistakes = [
        ("❌ 过早优化", "在需求明确前就优化"),
        ("❌ 盲目追求速度", "牺牲质量换速度"),
        ("❌ 忽略监控", "不知道瓶颈在哪"),
        ("❌ 配置不当", "batch size 过大导致 OOM"),
        ("❌ 不测试质量", "优化后质量下降")
    ]

    for mistake, desc in mistakes:
        console.print(f"{mistake}: {desc}")


def main():
    """主函数"""
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]推理优化示例[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    demos = [
        ("批处理优化", demo_batch_processing),
        ("KV Cache 优化", demo_kv_cache),
        ("Flash Attention", demo_flash_attention),
        ("并发优化", demo_concurrent_optimization),
        ("内存优化", demo_memory_optimization),
        ("优化工作流", demo_optimization_workflow),
        ("性能对比", demo_performance_comparison),
        ("最佳实践", demo_best_practices)
    ]

    console.print("\n[bold]选择要运行的演示:[/bold]")
    for i, (name, _) in enumerate(demos, 1):
        console.print(f"  {i}. {name}")
    console.print("  0. 运行所有演示")

    choice = input("\n请输入选项 (0-8): ").strip()

    if choice == "0":
        for name, func in demos:
            func()
    elif choice.isdigit() and 1 <= int(choice) <= len(demos):
        demos[int(choice) - 1][1]()
    else:
        console.print("[red]❌ 无效选项[/red]")


if __name__ == "__main__":
    main()
