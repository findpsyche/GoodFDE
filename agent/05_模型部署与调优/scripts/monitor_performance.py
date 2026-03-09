"""
性能监控脚本

实时监控模型推理性能
"""

import os
import time
import psutil
import requests
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel

load_dotenv()

console = Console()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host
        self.api_base = f"{host}/api"
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history = 100

    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # GPU 指标（如果有 nvidia-smi）
        gpu_info = self._get_gpu_info()

        return {
            "timestamp": datetime.now(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024 ** 3),
            "memory_total_gb": memory.total / (1024 ** 3),
            "gpu_info": gpu_info
        }

    def _get_gpu_info(self) -> Dict[str, Any]:
        """获取 GPU 信息"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                values = result.stdout.strip().split(',')
                return {
                    "gpu_utilization": float(values[0]),
                    "memory_used_mb": float(values[1]),
                    "memory_total_mb": float(values[2]),
                    "temperature": float(values[3])
                }
        except Exception:
            pass

        return None

    def test_inference(self, model: str = "llama3:8b") -> Dict[str, Any]:
        """测试推理性能"""
        try:
            data = {
                "model": model,
                "prompt": "What is AI?",
                "stream": False,
                "options": {"num_predict": 50}
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

                return {
                    "success": True,
                    "elapsed": elapsed,
                    "tokens": eval_count,
                    "speed": speed
                }
            else:
                return {"success": False}

        except Exception:
            return {"success": False}

    def generate_dashboard(self) -> Layout:
        """生成仪表板"""
        layout = Layout()

        # 系统指标
        system_metrics = self.get_system_metrics()

        system_table = Table(show_header=False, box=None)
        system_table.add_column("指标", style="cyan")
        system_table.add_column("数值", justify="right", style="green")

        system_table.add_row("CPU 使用率", f"{system_metrics['cpu_percent']:.1f}%")
        system_table.add_row("内存使用率", f"{system_metrics['memory_percent']:.1f}%")
        system_table.add_row(
            "内存使用",
            f"{system_metrics['memory_used_gb']:.1f} / {system_metrics['memory_total_gb']:.1f} GB"
        )

        if system_metrics['gpu_info']:
            gpu = system_metrics['gpu_info']
            system_table.add_row("GPU 使用率", f"{gpu['gpu_utilization']:.1f}%")
            system_table.add_row(
                "GPU 内存",
                f"{gpu['memory_used_mb']:.0f} / {gpu['memory_total_mb']:.0f} MB"
            )
            system_table.add_row("GPU 温度", f"{gpu['temperature']:.0f}°C")

        # 推理指标
        inference_table = Table(show_header=False, box=None)
        inference_table.add_column("指标", style="cyan")
        inference_table.add_column("数值", justify="right", style="green")

        if self.metrics_history:
            recent = [m for m in self.metrics_history[-10:] if m.get("success")]

            if recent:
                avg_speed = sum(m["speed"] for m in recent) / len(recent)
                avg_latency = sum(m["elapsed"] for m in recent) / len(recent)

                inference_table.add_row("平均速度", f"{avg_speed:.1f} tokens/s")
                inference_table.add_row("平均延迟", f"{avg_latency:.2f}s")
                inference_table.add_row("成功请求", f"{len(recent)}/10")

        # 组合布局
        layout.split_column(
            Layout(Panel(system_table, title="系统指标", border_style="cyan")),
            Layout(Panel(inference_table, title="推理指标", border_style="green"))
        )

        return layout

    def monitor(self, model: str = "llama3:8b", interval: int = 5):
        """开始监控"""
        console.print(f"[yellow]🔍 开始监控 {model}...[/yellow]")
        console.print(f"[dim]刷新间隔: {interval}秒[/dim]")
        console.print(f"[dim]按 Ctrl+C 停止[/dim]\n")

        with Live(self.generate_dashboard(), refresh_per_second=1) as live:
            try:
                while True:
                    # 测试推理
                    inference_result = self.test_inference(model)
                    self.metrics_history.append(inference_result)

                    # 限制历史记录数量
                    if len(self.metrics_history) > self.max_history:
                        self.metrics_history.pop(0)

                    # 更新仪表板
                    live.update(self.generate_dashboard())

                    time.sleep(interval)

            except KeyboardInterrupt:
                console.print("\n[yellow]⏹️  停止监控[/yellow]")


def main():
    """主函数"""
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]性能监控脚本[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print()

    monitor = PerformanceMonitor()

    # 获取模型列表
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                model = models[0].get("name")
                console.print(f"[green]使用模型: {model}[/green]\n")
                monitor.monitor(model, interval=5)
            else:
                console.print("[red]❌ 没有可用的模型[/red]")
        else:
            console.print("[red]❌ 无法获取模型列表[/red]")
    except Exception as e:
        console.print(f"[red]❌ 错误: {e}[/red]")


if __name__ == "__main__":
    main()
