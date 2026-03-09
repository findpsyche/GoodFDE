"""
模型对比脚本

对比不同模型或配置的性能
"""

import os
import time
import requests
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn

load_dotenv()

console = Console()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


@dataclass
class ComparisonResult:
    """对比结果"""
    model_name: str
    avg_speed: float
    avg_latency: float
    min_latency: float
    max_latency: float
    success_rate: float


class ModelComparator:
    """模型对比器"""

    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host
        self.api_base = f"{host}/api"

    def test_model(
        self,
        model: str,
        prompts: List[str],
        max_tokens: int = 100
    ) -> ComparisonResult:
        """测试单个模型"""
        console.print(f"[yellow]测试 {model}...[/yellow]")

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

            for prompt in prompts:
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
                        timeout=60
                    )
                    elapsed = time.time() - start_time

                    if response.status_code == 200:
                        result = response.json()
                        eval_count = result.get("eval_count", 0)
                        eval_duration = result.get("eval_duration", 0) / 1e9

                        speed = eval_count / eval_duration if eval_duration > 0 else 0

                        times.append(elapsed)
                        speeds.append(speed)
                    else:
                        errors += 1

                except Exception:
                    errors += 1

                progress.advance(task)

        # 计算统计数据
        if times:
            return ComparisonResult(
                model_name=model,
                avg_speed=statistics.mean(speeds),
                avg_latency=statistics.mean(times),
                min_latency=min(times),
                max_latency=max(times),
                success_rate=len(times) / len(prompts)
            )
        else:
            return None

    def compare_models(
        self,
        models: List[str],
        test_prompts: List[str]
    ) -> List[ComparisonResult]:
        """对比多个模型"""
        results = []

        for model in models:
            result = self.test_model(model, test_prompts)
            if result:
                results.append(result)

        return results

    def print_comparison(self, results: List[ComparisonResult]):
        """打印对比结果"""
        console.print("\n[bold cyan]📊 模型对比结果[/bold cyan]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("模型", style="cyan")
        table.add_column("平均速度", justify="right")
        table.add_column("平均延迟", justify="right")
        table.add_column("延迟范围", justify="right")
        table.add_column("成功率", justify="right")

        for r in results:
            table.add_row(
                r.model_name,
                f"{r.avg_speed:.1f} tok/s",
                f"{r.avg_latency:.2f}s",
                f"{r.min_latency:.2f}-{r.max_latency:.2f}s",
                f"{r.success_rate * 100:.1f}%"
            )

        console.print(table)

        # 找出最快的模型
        if results:
            fastest = max(results, key=lambda x: x.avg_speed)
            console.print(f"\n[green]🏆 最快模型: {fastest.model_name} ({fastest.avg_speed:.1f} tok/s)[/green]")


def main():
    """主函数"""
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]模型对比脚本[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print()

    comparator = ModelComparator()

    # 获取已安装的模型
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        if response.status_code == 200:
            all_models = response.json().get("models", [])
            model_names = [m.get("name") for m in all_models]

            if not model_names:
                console.print("[red]❌ 没有已安装的模型[/red]")
                return

            console.print(f"[green]找到 {len(model_names)} 个模型[/green]\n")

            # 选择要对比的模型
            console.print("[bold]可用模型:[/bold]")
            for i, name in enumerate(model_names, 1):
                console.print(f"  {i}. {name}")

            console.print("\n[dim]输入要对比的模型编号，用逗号分隔 (例如: 1,2,3)[/dim]")
            console.print("[dim]或按 Enter 对比所有模型[/dim]")

            choice = input("\n请选择: ").strip()

            if choice:
                indices = [int(x.strip()) - 1 for x in choice.split(",")]
                selected_models = [model_names[i] for i in indices if 0 <= i < len(model_names)]
            else:
                selected_models = model_names

            if not selected_models:
                console.print("[red]❌ 没有选择模型[/red]")
                return

            console.print(f"\n[yellow]将对比以下模型:[/yellow]")
            for model in selected_models:
                console.print(f"  • {model}")

            # 测试提示
            test_prompts = [
                "What is AI?",
                "Explain machine learning.",
                "What is deep learning?",
                "Define neural networks.",
                "What is NLP?"
            ]

            console.print(f"\n[yellow]使用 {len(test_prompts)} 个测试提示[/yellow]\n")

            # 开始对比
            results = comparator.compare_models(selected_models, test_prompts)

            # 打印结果
            if results:
                comparator.print_comparison(results)
            else:
                console.print("[red]❌ 对比失败[/red]")

        else:
            console.print("[red]❌ 无法获取模型列表[/red]")

    except Exception as e:
        console.print(f"[red]❌ 错误: {e}[/red]")


if __name__ == "__main__":
    main()
