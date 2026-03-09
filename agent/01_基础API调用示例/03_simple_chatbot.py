"""
简单的命令行聊天机器人

学习目标：
1. 实现一个可交互的聊天程序
2. 管理对话历史
3. 实现优雅的退出机制
"""

import os
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class SimpleChatbot:
    """简单的聊天机器人类"""

    def __init__(self, provider="openai", model=None):
        """
        初始化聊天机器人

        Args:
            provider: "openai" 或 "anthropic"
            model: 模型名称，如果为None则使用默认模型
        """
        self.provider = provider
        self.conversation_history = []

        if provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-3.5-turbo"
            self.system_message = {
                "role": "system",
                "content": "你是一个友好、有帮助的AI助手。"
            }
            self.conversation_history.append(self.system_message)

        elif provider == "anthropic":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model or "claude-3-5-haiku-20241022"
            self.system_prompt = "你是一个友好、有帮助的AI助手。"
            # Anthropic不在messages中包含system

        else:
            raise ValueError("provider必须是'openai'或'anthropic'")

    def chat(self, user_message):
        """
        发送消息并获取回复

        Args:
            user_message: 用户输入的消息

        Returns:
            AI的回复内容
        """
        # 添加用户消息到历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    temperature=0.7,
                    max_tokens=1000
                )
                assistant_message = response.choices[0].message.content

                # 添加助手回复到历史
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })

                return assistant_message

            else:  # anthropic
                # Anthropic需要分离system和messages
                messages = [msg for msg in self.conversation_history
                           if msg["role"] != "system"]

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    system=self.system_prompt,
                    messages=messages,
                    temperature=0.7
                )
                assistant_message = response.content[0].text

                # 添加助手回复到历史
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })

                return assistant_message

        except Exception as e:
            return f"❌ 错误: {str(e)}"

    def clear_history(self):
        """清空对话历史"""
        if self.provider == "openai":
            self.conversation_history = [self.system_message]
        else:
            self.conversation_history = []

    def get_history_length(self):
        """获取对话轮数"""
        # 不计算system message
        return len([msg for msg in self.conversation_history
                   if msg["role"] != "system"]) // 2


def print_welcome():
    """打印欢迎信息"""
    print("\n" + "=" * 60)
    print("🤖 欢迎使用简单聊天机器人！")
    print("=" * 60)
    print("\n命令说明:")
    print("  - 直接输入消息开始对话")
    print("  - 输入 'clear' 清空对话历史")
    print("  - 输入 'history' 查看对话轮数")
    print("  - 输入 'switch' 切换AI提供商")
    print("  - 输入 'quit' 或 'exit' 退出程序")
    print("\n" + "-" * 60 + "\n")


def main():
    """主函数"""
    print_welcome()

    # 选择AI提供商
    print("请选择AI提供商:")
    print("1. OpenAI (GPT-3.5)")
    print("2. Anthropic (Claude)")

    while True:
        choice = input("\n请输入选项 (1/2): ").strip()
        if choice == "1":
            provider = "openai"
            break
        elif choice == "2":
            provider = "anthropic"
            break
        else:
            print("无效选项，请重新输入")

    # 初始化聊天机器人
    try:
        bot = SimpleChatbot(provider=provider)
        print(f"\n✅ 已连接到 {provider.upper()}")
        print(f"📝 使用模型: {bot.model}\n")
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        print("请检查API密钥配置")
        return

    # 主循环
    while True:
        try:
            # 获取用户输入
            user_input = input("你: ").strip()

            # 处理特殊命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 再见！")
                break

            elif user_input.lower() == 'clear':
                bot.clear_history()
                print("\n✅ 对话历史已清空\n")
                continue

            elif user_input.lower() == 'history':
                count = bot.get_history_length()
                print(f"\n📊 当前对话轮数: {count}\n")
                continue

            elif user_input.lower() == 'switch':
                # 切换提供商
                new_provider = "anthropic" if provider == "openai" else "openai"
                try:
                    bot = SimpleChatbot(provider=new_provider)
                    provider = new_provider
                    print(f"\n✅ 已切换到 {provider.upper()}")
                    print(f"📝 使用模型: {bot.model}\n")
                except Exception as e:
                    print(f"\n❌ 切换失败: {e}\n")
                continue

            elif not user_input:
                continue

            # 发送消息并获取回复
            print("\n🤖 AI: ", end="", flush=True)
            response = bot.chat(user_input)
            print(response + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 检测到中断，正在退出...")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}\n")


if __name__ == "__main__":
    # 检查API密钥
    load_dotenv()

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if not has_openai and not has_anthropic:
        print("\n❌ 错误: 未找到任何API密钥")
        print("请在.env文件中设置OPENAI_API_KEY或ANTHROPIC_API_KEY")
        exit(1)

    if not has_openai:
        print("\n⚠️  警告: 未找到OPENAI_API_KEY，只能使用Anthropic")

    if not has_anthropic:
        print("\n⚠️  警告: 未找到ANTHROPIC_API_KEY，只能使用OpenAI")

    main()
