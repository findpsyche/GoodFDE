"""
Ollama 部署示例

本示例展示如何：
1. 检查 Ollama 服务状态
2. 管理模型（下载、列表、删除）
3. 进行基本推理
4. 实现流式响应
5. 创建自定义 Modelfile
6. 性能监控和日志记录

学习目标：
- 掌握 Ollama 的基本操作
- 理解 Modelfile 的配置
- 学会监控和调试
"""

import requests
import json
import time
import os
from typing import Optional, Dict, Any, Generator
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# 加载环境变量
load_dotenv()

# 配置
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
console = Console()


class OllamaClient:
    """Ollama 客户端封装"""

    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host
        self.api_base = f"{host}/api"

    def check_health(self) -> bool:
        """检查 Ollama 服务是否运行"""
        try:
            response = requests.get(f"{self.host}/", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> list:
        """列出已安装的模型"""
        try:
            response = requests.get(f"{self.api_base}/tags")
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            return []
        except requests.exceptions.RequestException as e:
            console.print(f"[red]❌ 获取模型列表失败: {e}[/red]")
            return []

    def pull_model(self, model_name: str) -> bool:
        """下载模型"""
        console.print(f"[yellow]📥 开始下载模型: {model_name}[/yellow]")

        try:
            response = requests.post(
                f"{self.api_base}/pull",
                json={"name": model_name},
                stream=True,
                timeout=3600  # 1小时超时
            )

            if response.status_code == 200:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("下载中...", total=None)

                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            status = data.get("status", "")
                            progress.update(task, description=status)

                            if "error" in data:
                                console.print(f"[red]❌ 错误: {data['error']}[/red]")
                                return False

                console.print(f"[green]✅ 模型下载完成: {model_name}[/green]")
                return True
            else:
                console.print(f"[red]❌ 下载失败: {response.status_code}[/red]")
                return False

        except requests.exceptions.RequestException as e:
            console.print(f"[red]❌ 下载失败: {e}[/red]")
            return False

    def delete_model(self, model_name: str) -> bool:
        """删除模型"""
        try:
            response = requests.delete(
                f"{self.api_base}/delete",
                json={"name": model_name}
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """生成响应（非流式）"""
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }

        if options:
            data["options"] = options

        try:
            response = requests.post(
                f"{self.api_base}/generate",
                json=data,
                timeout=300
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}

        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def generate_stream(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """生成响应（流式）"""
        data = {
            "model": model,
            "prompt": prompt,
            "stream": True
        }

        if options:
            data["options"] = options

        try:
            response = requests.post(
                f"{self.api_base}/generate",
                json=data,
                stream=True,
                timeout=300
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line)
            else:
                yield {"error": f"HTTP {response.status_code}"}

        except requests.exceptions.RequestException as e:
            yield {"error": str(e)}

    def create_model(self, name: str, modelfile: str) -> bool:
        """从 Modelfile 创建自定义模型"""
        try:
            response = requests.post(
                f"{self.api_base}/create",
                json={"name": name, "modelfile": modelfile},
                stream=True
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "error" in data:
                            console.print(f"[red]❌ 错误: {data['error']}[/red]")
                            return False
                return True
            return False

        except requests.exceptions.RequestException:
            return False


def demo_basic_usage():
    """演示基本使用"""
    console.print("\n[bold cyan]🚀 演示1: 基本使用[/bold cyan]\n")

    client = OllamaClient()

    # 1. 检查服务状态
    console.print("[yellow]1. 检查 Ollama 服务状态...[/yellow]")
    if client.check_health():
        console.print("[green]✅ Ollama 服务正常运行[/green]")
    else:
        console.print("[red]❌ Ollama 服务未运行，请先启动: ollama serve[/red]")
        return

    # 2. 列出已安装的模型
    console.print("\n[yellow]2. 已安装的模型:[/yellow]")
    models = client.list_models()

    if models:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("模型名称", style="cyan")
        table.add_column("大小", justify="right")
        table.add_column("修改时间")

        for model in models:
            name = model.get("name", "")
            size = model.get("size", 0)
            size_gb = size / (1024 ** 3)
            modified = model.get("modified_at", "")

            table.add_row(name, f"{size_gb:.2f} GB", modified[:10])

        console.print(table)
    else:
        console.print("[yellow]⚠️  没有已安装的模型[/yellow]")

    # 3. 基本推理测试
    model_name = "llama3:8b"

    # 检查模型是否存在
    model_exists = any(m.get("name") == model_name for m in models)

    if not model_exists:
        console.print(f"\n[yellow]3. 模型 {model_name} 未安装，是否下载？(y/n)[/yellow]")
        choice = input().strip().lower()
        if choice == 'y':
            client.pull_model(model_name)
        else:
            console.print("[yellow]⚠️  跳过推理测试[/yellow]")
            return

    console.print(f"\n[yellow]3. 测试基本推理 ({model_name})...[/yellow]")

    prompt = "What is the capital of France? Answer in one sentence."

    start_time = time.time()
    result = client.generate(model_name, prompt)
    elapsed = time.time() - start_time

    if "error" in result:
        console.print(f"[red]❌ 推理失败: {result['error']}[/red]")
    else:
        response_text = result.get("response", "")
        total_duration = result.get("total_duration", 0) / 1e9  # 纳秒转秒

        console.print(f"\n[green]✅ 推理成功！[/green]")
        console.print(f"[cyan]问题:[/cyan] {prompt}")
        console.print(f"[cyan]回答:[/cyan] {response_text}")
        console.print(f"\n[dim]⏱️  总耗时: {elapsed:.2f}s[/dim]")
        console.print(f"[dim]⏱️  模型耗时: {total_duration:.2f}s[/dim]")

        # 计算速度
        if "eval_count" in result and "eval_duration" in result:
            tokens = result["eval_count"]
            duration = result["eval_duration"] / 1e9
            speed = tokens / duration if duration > 0 else 0
            console.print(f"[dim]🚀 生成速度: {speed:.1f} tokens/s[/dim]")


def demo_streaming():
    """演示流式响应"""
    console.print("\n[bold cyan]🚀 演示2: 流式响应[/bold cyan]\n")

    client = OllamaClient()
    model_name = "llama3:8b"

    prompt = "Write a short poem about artificial intelligence."

    console.print(f"[cyan]问题:[/cyan] {prompt}")
    console.print(f"[cyan]回答:[/cyan] ", end="")

    start_time = time.time()
    token_count = 0

    for chunk in client.generate_stream(model_name, prompt):
        if "error" in chunk:
            console.print(f"\n[red]❌ 错误: {chunk['error']}[/red]")
            break

        if "response" in chunk:
            text = chunk["response"]
            console.print(text, end="", style="green")
            token_count += 1

        if chunk.get("done", False):
            elapsed = time.time() - start_time
            console.print(f"\n\n[dim]⏱️  耗时: {elapsed:.2f}s[/dim]")
            console.print(f"[dim]🚀 速度: {token_count/elapsed:.1f} tokens/s[/dim]")
            break


def demo_custom_parameters():
    """演示自定义参数"""
    console.print("\n[bold cyan]🚀 演示3: 自定义参数[/bold cyan]\n")

    client = OllamaClient()
    model_name = "llama3:8b"

    prompt = "Tell me a creative story about a robot."

    # 测试不同的温度参数
    temperatures = [0.1, 0.7, 1.5]

    for temp in temperatures:
        console.print(f"\n[yellow]Temperature = {temp}[/yellow]")

        options = {
            "temperature": temp,
            "top_p": 0.9,
            "top_k": 40
        }

        result = client.generate(model_name, prompt, options=options)

        if "error" not in result:
            response = result.get("response", "")
            console.print(f"[green]{response[:200]}...[/green]")
        else:
            console.print(f"[red]❌ 错误: {result['error']}[/red]")


def demo_custom_modelfile():
    """演示自定义 Modelfile"""
    console.print("\n[bold cyan]🚀 演示4: 自定义 Modelfile[/bold cyan]\n")

    client = OllamaClient()

    # 创建自定义 Modelfile
    modelfile = """
FROM llama3:8b

# 设置温度
PARAMETER temperature 0.8

# 设置系统提示词
SYSTEM """
你是一个友好的 AI 助手，专门帮助用户学习编程。
你的回答应该：
1. 简洁明了
2. 包含代码示例
3. 解释关键概念
"""

# 设置停止词
PARAMETER stop "```"
"""

    custom_model_name = "my-coding-assistant"

    console.print(f"[yellow]创建自定义模型: {custom_model_name}[/yellow]")
    console.print(f"\n[dim]Modelfile:[/dim]")
    console.print(modelfile)

    if client.create_model(custom_model_name, modelfile):
        console.print(f"\n[green]✅ 自定义模型创建成功！[/green]")

        # 测试自定义模型
        console.print(f"\n[yellow]测试自定义模型...[/yellow]")
        prompt = "How do I create a list in Python?"

        result = client.generate(custom_model_name, prompt)

        if "error" not in result:
            response = result.get("response", "")
            console.print(f"\n[cyan]问题:[/cyan] {prompt}")
            console.print(f"[cyan]回答:[/cyan] {response}")
        else:
            console.print(f"[red]❌ 错误: {result['error']}[/red]")
    else:
        console.print(f"[red]❌ 自定义模型创建失败[/red]")


def demo_performance_monitoring():
    """演示性能监控"""
    console.print("\n[bold cyan]🚀 演示5: 性能监控[/bold cyan]\n")

    client = OllamaClient()
    model_name = "llama3:8b"

    prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "What is deep learning?",
        "Define artificial intelligence.",
        "What is natural language processing?"
    ]

    results = []

    console.print("[yellow]运行性能测试...[/yellow]\n")

    for i, prompt in enumerate(prompts, 1):
        console.print(f"[dim]测试 {i}/{len(prompts)}...[/dim]", end=" ")

        start_time = time.time()
        result = client.generate(model_name, prompt)
        elapsed = time.time() - start_time

        if "error" not in result:
            tokens = result.get("eval_count", 0)
            speed = tokens / elapsed if elapsed > 0 else 0

            results.append({
                "prompt": prompt,
                "elapsed": elapsed,
                "tokens": tokens,
                "speed": speed
            })

            console.print(f"[green]✅ {elapsed:.2f}s, {speed:.1f} tokens/s[/green]")
        else:
            console.print(f"[red]❌ 失败[/red]")

    # 显示统计结果
    if results:
        console.print("\n[bold]📊 性能统计:[/bold]")

        avg_time = sum(r["elapsed"] for r in results) / len(results)
        avg_speed = sum(r["speed"] for r in results) / len(results)
        max_speed = max(r["speed"] for r in results)
        min_speed = min(r["speed"] for r in results)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan")
        table.add_column("数值", justify="right", style="green")

        table.add_row("平均耗时", f"{avg_time:.2f}s")
        table.add_row("平均速度", f"{avg_speed:.1f} tokens/s")
        table.add_row("最快速度", f"{max_speed:.1f} tokens/s")
        table.add_row("最慢速度", f"{min_speed:.1f} tokens/s")

        console.print(table)


def main():
    """主函数"""
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]Ollama 部署示例[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    demos = [
        ("基本使用", demo_basic_usage),
        ("流式响应", demo_streaming),
        ("自定义参数", demo_custom_parameters),
        ("自定义 Modelfile", demo_custom_modelfile),
        ("性能监控", demo_performance_monitoring)
    ]

    console.print("\n[bold]选择要运行的演示:[/bold]")
    for i, (name, _) in enumerate(demos, 1):
        console.print(f"  {i}. {name}")
    console.print("  0. 运行所有演示")

    choice = input("\n请输入选项 (0-5): ").strip()

    if choice == "0":
        for name, func in demos:
            func()
    elif choice.isdigit() and 1 <= int(choice) <= len(demos):
        demos[int(choice) - 1][1]()
    else:
        console.print("[red]❌ 无效选项[/red]")


if __name__ == "__main__":
    main()
