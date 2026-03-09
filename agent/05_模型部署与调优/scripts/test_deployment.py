"""
部署测试脚本

测试模型部署是否正常工作
"""

import os
import sys
import time
import requests
from typing import Dict, Any, List
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

console = Console()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


class DeploymentTester:
    """部署测试器"""

    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host
        self.api_base = f"{host}/api"
        self.test_results = []

    def test_service_health(self) -> bool:
        """测试服务健康状态"""
        console.print("[yellow]1. 测试服务健康状态...[/yellow]")

        try:
            response = requests.get(f"{self.host}/", timeout=5)
            if response.status_code == 200:
                console.print("[green]✅ 服务正常运行[/green]")
                self.test_results.append(("服务健康", "通过"))
                return True
            else:
                console.print(f"[red]❌ 服务异常: HTTP {response.status_code}[/red]")
                self.test_results.append(("服务健康", "失败"))
                return False
        except Exception as e:
            console.print(f"[red]❌ 无法连接服务: {e}[/red]")
            self.test_results.append(("服务健康", "失败"))
            return False

    def test_list_models(self) -> bool:
        """测试列出模型"""
        console.print("\n[yellow]2. 测试列出模型...[/yellow]")

        try:
            response = requests.get(f"{self.api_base}/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])

                if models:
                    console.print(f"[green]✅ 找到 {len(models)} 个模型[/green]")
                    for model in models[:3]:  # 只显示前3个
                        console.print(f"  - {model.get('name')}")
                    self.test_results.append(("列出模型", "通过"))
                    return True
                else:
                    console.print("[yellow]⚠️  没有已安装的模型[/yellow]")
                    self.test_results.append(("列出模型", "警告"))
                    return False
            else:
                console.print(f"[red]❌ 获取模型列表失败: HTTP {response.status_code}[/red]")
                self.test_results.append(("列出模型", "失败"))
                return False
        except Exception as e:
            console.print(f"[red]❌ 错误: {e}[/red]")
            self.test_results.append(("列出模型", "失败"))
            return False

    def test_basic_inference(self, model: str = "llama3:8b") -> bool:
        """测试基本推理"""
        console.print(f"\n[yellow]3. 测试基本推理 ({model})...[/yellow]")

        try:
            data = {
                "model": model,
                "prompt": "Say 'Hello, World!' and nothing else.",
                "stream": False,
                "options": {"num_predict": 10}
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
                output = result.get("response", "")

                console.print(f"[green]✅ 推理成功[/green]")
                console.print(f"[dim]输出: {output[:100]}...[/dim]")
                console.print(f"[dim]耗时: {elapsed:.2f}s[/dim]")

                self.test_results.append(("基本推理", "通过"))
                return True
            else:
                console.print(f"[red]❌ 推理失败: HTTP {response.status_code}[/red]")
                self.test_results.append(("基本推理", "失败"))
                return False
        except Exception as e:
            console.print(f"[red]❌ 错误: {e}[/red]")
            self.test_results.append(("基本推理", "失败"))
            return False

    def test_streaming(self, model: str = "llama3:8b") -> bool:
        """测试流式响应"""
        console.print(f"\n[yellow]4. 测试流式响应 ({model})...[/yellow]")

        try:
            data = {
                "model": model,
                "prompt": "Count from 1 to 5.",
                "stream": True,
                "options": {"num_predict": 20}
            }

            response = requests.post(
                f"{self.api_base}/generate",
                json=data,
                stream=True,
                timeout=60
            )

            if response.status_code == 200:
                chunks = 0
                for line in response.iter_lines():
                    if line:
                        chunks += 1
                        if chunks > 10:  # 只测试前10个chunk
                            break

                console.print(f"[green]✅ 流式响应正常 (收到 {chunks} 个chunk)[/green]")
                self.test_results.append(("流式响应", "通过"))
                return True
            else:
                console.print(f"[red]❌ 流式响应失败: HTTP {response.status_code}[/red]")
                self.test_results.append(("流式响应", "失败"))
                return False
        except Exception as e:
            console.print(f"[red]❌ 错误: {e}[/red]")
            self.test_results.append(("流式响应", "失败"))
            return False

    def test_performance(self, model: str = "llama3:8b") -> bool:
        """测试性能"""
        console.print(f"\n[yellow]5. 测试性能 ({model})...[/yellow]")

        try:
            data = {
                "model": model,
                "prompt": "What is AI?",
                "stream": False,
                "options": {"num_predict": 50}
            }

            times = []
            speeds = []

            for i in range(3):
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

            if times:
                avg_time = sum(times) / len(times)
                avg_speed = sum(speeds) / len(speeds)

                console.print(f"[green]✅ 性能测试完成[/green]")
                console.print(f"[dim]平均耗时: {avg_time:.2f}s[/dim]")
                console.print(f"[dim]平均速度: {avg_speed:.1f} tokens/s[/dim]")

                # 性能判断
                if avg_speed > 20:
                    self.test_results.append(("性能测试", "通过"))
                    return True
                else:
                    console.print("[yellow]⚠️  性能较低 (< 20 tokens/s)[/yellow]")
                    self.test_results.append(("性能测试", "警告"))
                    return False
            else:
                console.print("[red]❌ 性能测试失败[/red]")
                self.test_results.append(("性能测试", "失败"))
                return False

        except Exception as e:
            console.print(f"[red]❌ 错误: {e}[/red]")
            self.test_results.append(("性能测试", "失败"))
            return False

    def print_summary(self):
        """打印测试摘要"""
        console.print("\n[bold cyan]📊 测试摘要[/bold cyan]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("测试项", style="cyan")
        table.add_column("结果", justify="center")

        for test_name, result in self.test_results:
            if result == "通过":
                style = "green"
                symbol = "✅"
            elif result == "警告":
                style = "yellow"
                symbol = "⚠️"
            else:
                style = "red"
                symbol = "❌"

            table.add_row(test_name, f"[{style}]{symbol} {result}[/{style}]")

        console.print(table)

        # 统计
        passed = sum(1 for _, r in self.test_results if r == "通过")
        total = len(self.test_results)

        console.print(f"\n[bold]通过率: {passed}/{total} ({passed/total*100:.1f}%)[/bold]")

        if passed == total:
            console.print("\n[green]🎉 所有测试通过！部署正常。[/green]")
            return 0
        else:
            console.print("\n[yellow]⚠️  部分测试未通过，请检查配置。[/yellow]")
            return 1


def main():
    """主函数"""
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]部署测试脚本[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print()

    tester = DeploymentTester()

    # 运行测试
    tester.test_service_health()
    tester.test_list_models()
    tester.test_basic_inference()
    tester.test_streaming()
    tester.test_performance()

    # 打印摘要
    exit_code = tester.print_summary()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
