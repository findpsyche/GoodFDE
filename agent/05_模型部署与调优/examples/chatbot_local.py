"""
本地聊天机器人示例

一个完整的本地聊天机器人实现，包括：
- 对话历史管理
- 流式响应
- 上下文窗口管理
- 性能监控
"""

import os
import time
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv()

console = Console()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")


@dataclass
class Message:
    """消息"""
    role: str  # system, user, assistant
    content: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ChatBot:
    """本地聊天机器人"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system_prompt: Optional[str] = None,
        max_history: int = 10,
        max_context_length: int = 4096
    ):
        self.model = model
        self.system_prompt = system_prompt or "你是一个友好的AI助手。"
        self.max_history = max_history
        self.max_context_length = max_context_length
        self.history: List[Message] = []
        self.api_base = f"{OLLAMA_HOST}/api"

        # 添加系统消息
        self.history.append(Message(role="system", content=self.system_prompt))

    def _build_prompt(self) -> str:
        """构建完整的 prompt"""
        # 保留最近的 N 条消息
        recent_history = self.history[-self.max_history:]

        prompt_parts = []
        for msg in recent_history:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt_parts.append("Assistant:")

        return "\n\n".join(prompt_parts)

    def chat(self, user_input: str, stream: bool = True) -> str:
        """发送消息并获取响应"""
        # 添加用户消息
        self.history.append(Message(role="user", content=user_input))

        # 构建 prompt
        prompt = self._build_prompt()

        # 调用 API
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream
        }

        try:
            response = requests.post(
                f"{self.api_base}/generate",
                json=data,
                stream=stream,
                timeout=300
            )

            if stream:
                # 流式响应
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        chunk = line.decode('utf-8')
                        import json
                        data = json.loads(chunk)

                        if "response" in data:
                            text = data["response"]
                            console.print(text, end="", style="green")
                            full_response += text

                        if data.get("done", False):
                            break

                console.print()  # 换行

                # 添加助手消息
                self.history.append(Message(role="assistant", content=full_response))

                return full_response
            else:
                # 非流式响应
                result = response.json()
                assistant_response = result.get("response", "")

                # 添加助手消息
                self.history.append(Message(role="assistant", content=assistant_response))

                return assistant_response

        except Exception as e:
            error_msg = f"错误: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg

    def clear_history(self):
        """清除对话历史"""
        self.history = [Message(role="system", content=self.system_prompt)]

    def get_history(self) -> List[Message]:
        """获取对话历史"""
        return self.history

    def save_history(self, filename: str):
        """保存对话历史"""
        import json

        history_data = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in self.history
        ]

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)

        console.print(f"[green]✅ 对话历史已保存到: {filename}[/green]")


def main():
    """主函数"""
    console.print(Panel.fit(
        "[bold cyan]🤖 本地聊天机器人[/bold cyan]\n\n"
        "命令:\n"
        "  /help - 显示帮助\n"
        "  /clear - 清除对话历史\n"
        "  /history - 显示对话历史\n"
        "  /save - 保存对话历史\n"
        "  /quit - 退出",
        title="欢迎",
        border_style="cyan"
    ))

    # 初始化聊天机器人
    chatbot = ChatBot(
        model=DEFAULT_MODEL,
        system_prompt="你是一个友好、专业的AI助手。请用简洁、准确的语言回答问题。",
        max_history=10
    )

    console.print(f"\n[dim]使用模型: {DEFAULT_MODEL}[/dim]")
    console.print(f"[dim]最大历史: {chatbot.max_history} 条消息[/dim]\n")

    while True:
        try:
            # 获取用户输入
            user_input = console.input("[bold blue]你:[/bold blue] ").strip()

            if not user_input:
                continue

            # 处理命令
            if user_input.startswith("/"):
                command = user_input[1:].lower()

                if command == "quit" or command == "exit":
                    console.print("\n[yellow]👋 再见！[/yellow]")
                    break

                elif command == "help":
                    console.print("\n[bold]可用命令:[/bold]")
                    console.print("  /help - 显示此帮助")
                    console.print("  /clear - 清除对话历史")
                    console.print("  /history - 显示对话历史")
                    console.print("  /save - 保存对话历史")
                    console.print("  /quit - 退出\n")
                    continue

                elif command == "clear":
                    chatbot.clear_history()
                    console.print("[green]✅ 对话历史已清除[/green]\n")
                    continue

                elif command == "history":
                    console.print("\n[bold]对话历史:[/bold]\n")
                    for msg in chatbot.get_history():
                        if msg.role == "system":
                            console.print(f"[dim]System: {msg.content}[/dim]")
                        elif msg.role == "user":
                            console.print(f"[blue]User: {msg.content}[/blue]")
                        elif msg.role == "assistant":
                            console.print(f"[green]Assistant: {msg.content[:100]}...[/green]")
                    console.print()
                    continue

                elif command == "save":
                    filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    chatbot.save_history(filename)
                    continue

                else:
                    console.print(f"[red]❌ 未知命令: {command}[/red]\n")
                    continue

            # 发送消息
            console.print("[bold green]🤖:[/bold green] ", end="")

            start_time = time.time()
            response = chatbot.chat(user_input, stream=True)
            elapsed = time.time() - start_time

            console.print(f"\n[dim]⏱️  {elapsed:.2f}s[/dim]\n")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]👋 再见！[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]❌ 错误: {e}[/red]\n")


if __name__ == "__main__":
    main()
