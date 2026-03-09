"""
LoRA 微调示例

本示例展示如何：
1. 准备微调数据
2. 配置 LoRA 参数
3. 训练 LoRA 模型
4. 评估微调效果
5. 合并和部署模型

学习目标：
- 理解 LoRA 的原理和优势
- 掌握数据准备流程
- 学会训练和评估
- 理解何时使用微调

注意：需要 GPU 支持
"""

import os
import json
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()

console = Console()

# 检查依赖
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType
    )
    from datasets import Dataset
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    console.print("[yellow]⚠️  缺少依赖。请运行:[/yellow]")
    console.print("pip install transformers peft datasets accelerate bitsandbytes")


@dataclass
class FineTuningConfig:
    """微调配置"""
    model_name: str = "meta-llama/Llama-2-7b-hf"
    output_dir: str = "./outputs/lora"

    # LoRA 参数
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = None

    # 训练参数
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_length: int = 512

    # 量化
    use_4bit: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class LoRATrainer:
    """LoRA 训练器"""

    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """加载模型和分词器"""
        console.print(f"[yellow]📥 加载模型: {self.config.model_name}[/yellow]")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型（4-bit 量化）
        if self.config.use_4bit:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )

            # 准备模型用于训练
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                trust_remote_code=True
            )

        console.print("[green]✅ 模型加载完成[/green]")

    def setup_lora(self):
        """配置 LoRA"""
        console.print("[yellow]⚙️  配置 LoRA...[/yellow]")

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(self.model, lora_config)

        # 打印可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        console.print(f"[green]✅ LoRA 配置完成[/green]")
        console.print(f"[dim]可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)[/dim]")
        console.print(f"[dim]总参数: {total_params:,}[/dim]")

    def prepare_data(self, data: List[Dict[str, str]]) -> Dataset:
        """准备训练数据"""
        console.print(f"[yellow]📊 准备数据 ({len(data)} 条)...[/yellow]")

        # 格式化数据
        formatted_data = []
        for item in data:
            # Instruction format
            text = f"### Instruction:\n{item['instruction']}\n\n"
            if item.get('input'):
                text += f"### Input:\n{item['input']}\n\n"
            text += f"### Response:\n{item['output']}"

            formatted_data.append({"text": text})

        # 创建 Dataset
        dataset = Dataset.from_list(formatted_data)

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        console.print(f"[green]✅ 数据准备完成[/green]")
        return tokenized_dataset

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """训练模型"""
        console.print("[yellow]🚀 开始训练...[/yellow]")

        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=self.config.learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            report_to="none"  # 不使用 wandb
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

        # 训练
        trainer.train()

        console.print("[green]✅ 训练完成[/green]")

        # 保存模型
        trainer.save_model()
        console.print(f"[green]✅ 模型已保存到: {self.config.output_dir}[/green]")

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """生成响应"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 移除输入部分
        response = response[len(prompt):].strip()

        return response


def demo_data_preparation():
    """演示数据准备"""
    console.print("\n[bold cyan]🚀 演示1: 数据准备[/bold cyan]\n")

    console.print("[yellow]1. 数据格式[/yellow]\n")

    # 示例数据
    example_data = [
        {
            "instruction": "将以下 Python 代码添加注释",
            "input": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",
            "output": "def factorial(n):\n    \"\"\"计算阶乘\"\"\"\n    # 基础情况：0的阶乘是1\n    if n == 0:\n        return 1\n    # 递归情况：n! = n * (n-1)!\n    return n * factorial(n-1)"
        },
        {
            "instruction": "生成 SQL 查询",
            "input": "查询所有年龄大于25岁的用户",
            "output": "SELECT * FROM users WHERE age > 25;"
        },
        {
            "instruction": "解释概念",
            "input": "什么是机器学习？",
            "output": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并改进性能，而无需明确编程。"
        }
    ]

    console.print("[bold]示例数据:[/bold]\n")

    for i, item in enumerate(example_data, 1):
        console.print(f"[yellow]样本 {i}:[/yellow]")
        console.print(f"[cyan]指令:[/cyan] {item['instruction']}")
        if item.get('input'):
            console.print(f"[cyan]输入:[/cyan] {item['input'][:50]}...")
        console.print(f"[cyan]输出:[/cyan] {item['output'][:50]}...\n")

    console.print("[yellow]2. 数据质量检查[/yellow]\n")

    quality_checks = [
        ("✅ 格式统一", "所有数据使用相同的格式"),
        ("✅ 输入输出对应", "每个输入都有正确的输出"),
        ("✅ 长度适中", "避免过长或过短的样本"),
        ("✅ 多样性", "覆盖不同类型的任务"),
        ("✅ 质量高", "输出准确、完整、规范")
    ]

    for check, desc in quality_checks:
        console.print(f"{check}: {desc}")

    console.print("\n[yellow]3. 数据量建议[/yellow]\n")

    data_size_table = Table(show_header=True, header_style="bold magenta")
    data_size_table.add_column("任务类型")
    data_size_table.add_column("最小数据量", justify="right")
    data_size_table.add_column("推荐数据量", justify="right")
    data_size_table.add_column("备注")

    data_size_table.add_row("简单分类", "50-100", "200-500", "类别明确")
    data_size_table.add_row("文本生成", "100-200", "500-1000", "需要多样性")
    data_size_table.add_row("代码生成", "200-500", "1000-2000", "覆盖多种场景")
    data_size_table.add_row("领域知识", "500-1000", "2000-5000", "知识密集")

    console.print(data_size_table)


def demo_lora_configuration():
    """演示 LoRA 配置"""
    console.print("\n[bold cyan]🚀 演示2: LoRA 配置[/bold cyan]\n")

    console.print("[yellow]1. LoRA 参数说明[/yellow]\n")

    param_table = Table(show_header=True, header_style="bold magenta")
    param_table.add_column("参数", style="cyan")
    param_table.add_column("说明")
    param_table.add_column("典型值")
    param_table.add_column("影响")

    param_table.add_row(
        "r (rank)",
        "LoRA 矩阵的秩",
        "8, 16, 32",
        "越大越灵活，但训练慢"
    )
    param_table.add_row(
        "alpha",
        "缩放因子",
        "16, 32",
        "通常设为 r 的 2 倍"
    )
    param_table.add_row(
        "dropout",
        "Dropout 比率",
        "0.05, 0.1",
        "防止过拟合"
    )
    param_table.add_row(
        "target_modules",
        "要应用 LoRA 的层",
        "q_proj, v_proj",
        "影响可训练参数量"
    )

    console.print(param_table)

    console.print("\n[yellow]2. 配置示例[/yellow]\n")

    configs = [
        {
            "name": "轻量配置",
            "r": 4,
            "alpha": 8,
            "target": ["q_proj"],
            "use_case": "快速实验、资源受限"
        },
        {
            "name": "标准配置",
            "r": 8,
            "alpha": 16,
            "target": ["q_proj", "v_proj"],
            "use_case": "通用推荐"
        },
        {
            "name": "高质量配置",
            "r": 16,
            "alpha": 32,
            "target": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "use_case": "质量优先、充足资源"
        }
    ]

    for config in configs:
        console.print(f"[bold]{config['name']}:[/bold]")
        console.print(f"  r={config['r']}, alpha={config['alpha']}")
        console.print(f"  target_modules={config['target']}")
        console.print(f"  [dim]适用: {config['use_case']}[/dim]\n")


def demo_training_process():
    """演示训练流程"""
    console.print("\n[bold cyan]🚀 演示3: 训练流程[/bold cyan]\n")

    if not DEPENDENCIES_AVAILABLE:
        console.print("[red]❌ 缺少依赖，跳过演示[/red]")
        return

    console.print("[yellow]训练流程概览:[/yellow]\n")

    steps = [
        "1. 准备数据 → 格式化、分词、验证",
        "2. 加载模型 → 基础模型 + 量化（可选）",
        "3. 配置 LoRA → 设置参数、应用到模型",
        "4. 训练 → 监控 loss、保存 checkpoint",
        "5. 评估 → 验证集测试、质量评估",
        "6. 合并 → 合并 LoRA 权重到基础模型",
        "7. 部署 → 导出、优化、上线"
    ]

    for step in steps:
        console.print(f"[green]{step}[/green]")

    console.print("\n[yellow]注意事项:[/yellow]\n")

    notes = [
        "⚠️  监控训练 loss，确保正常下降",
        "⚠️  定期在验证集上测试，防止过拟合",
        "⚠️  保存多个 checkpoint，以便回退",
        "⚠️  测试边界情况，确保泛化能力",
        "⚠️  对比微调前后的输出质量"
    ]

    for note in notes:
        console.print(note)


def demo_evaluation():
    """演示评估方法"""
    console.print("\n[bold cyan]🚀 演示4: 评估方法[/bold cyan]\n")

    console.print("[yellow]1. 定量评估[/yellow]\n")

    metrics_table = Table(show_header=True, header_style="bold magenta")
    metrics_table.add_column("指标", style="cyan")
    metrics_table.add_column("适用任务")
    metrics_table.add_column("说明")

    metrics_table.add_row(
        "Perplexity",
        "语言模型",
        "困惑度，越低越好"
    )
    metrics_table.add_row(
        "BLEU",
        "翻译、生成",
        "与参考答案的相似度"
    )
    metrics_table.add_row(
        "ROUGE",
        "摘要",
        "召回率导向的相似度"
    )
    metrics_table.add_row(
        "Accuracy",
        "分类",
        "准确率"
    )
    metrics_table.add_row(
        "F1 Score",
        "分类",
        "精确率和召回率的调和平均"
    )

    console.print(metrics_table)

    console.print("\n[yellow]2. 定性评估[/yellow]\n")

    qualitative_checks = [
        "✅ 输出是否流畅自然",
        "✅ 是否遵循指令",
        "✅ 是否包含必要信息",
        "✅ 是否有幻觉或错误",
        "✅ 格式是否正确"
    ]

    for check in qualitative_checks:
        console.print(check)

    console.print("\n[yellow]3. A/B 测试[/yellow]\n")

    console.print("[dim]对比微调前后的输出:[/dim]\n")

    example_comparison = [
        {
            "prompt": "解释什么是 LoRA",
            "before": "LoRA 是一种技术...",
            "after": "LoRA (Low-Rank Adaptation) 是一种高效的模型微调方法，通过在预训练模型的权重矩阵上添加低秩分解矩阵来实现适应..."
        }
    ]

    for comp in example_comparison:
        console.print(f"[cyan]提示:[/cyan] {comp['prompt']}")
        console.print(f"[yellow]微调前:[/yellow] {comp['before']}")
        console.print(f"[green]微调后:[/green] {comp['after']}\n")


def demo_deployment():
    """演示部署流程"""
    console.print("\n[bold cyan]🚀 演示5: 部署流程[/bold cyan]\n")

    console.print("[yellow]1. 合并 LoRA 权重[/yellow]\n")

    console.print("[dim]方法1: 使用 PEFT 合并[/dim]")
    console.print("```python")
    console.print("from peft import PeftModel")
    console.print("")
    console.print("# 加载基础模型")
    console.print("base_model = AutoModelForCausalLM.from_pretrained('base_model')")
    console.print("")
    console.print("# 加载 LoRA 适配器")
    console.print("model = PeftModel.from_pretrained(base_model, 'lora_checkpoint')")
    console.print("")
    console.print("# 合并权重")
    console.print("merged_model = model.merge_and_unload()")
    console.print("")
    console.print("# 保存")
    console.print("merged_model.save_pretrained('merged_model')")
    console.print("```\n")

    console.print("[yellow]2. 导出为 GGUF（用于 Ollama）[/yellow]\n")

    console.print("[dim]步骤:[/dim]")
    console.print("1. 合并 LoRA 权重")
    console.print("2. 使用 llama.cpp 转换为 GGUF")
    console.print("3. 量化（可选）")
    console.print("4. 导入到 Ollama\n")

    console.print("[yellow]3. 部署选项[/yellow]\n")

    deployment_table = Table(show_header=True, header_style="bold magenta")
    deployment_table.add_column("方案", style="cyan")
    deployment_table.add_column("优点")
    deployment_table.add_column("缺点")
    deployment_table.add_column("适用场景")

    deployment_table.add_row(
        "Ollama",
        "易用、快速",
        "性能一般",
        "开发测试"
    )
    deployment_table.add_row(
        "vLLM",
        "高性能",
        "配置复杂",
        "生产环境"
    )
    deployment_table.add_row(
        "HF Inference",
        "托管服务",
        "成本高",
        "快速上线"
    )
    deployment_table.add_row(
        "自建服务",
        "完全控制",
        "维护成本高",
        "大规模应用"
    )

    console.print(deployment_table)


def demo_best_practices():
    """演示最佳实践"""
    console.print("\n[bold cyan]🚀 演示6: 最佳实践[/bold cyan]\n")

    console.print("[yellow]1. 何时使用微调[/yellow]\n")

    use_cases = [
        ("✅ 特定领域知识", "医疗、法律等专业领域"),
        ("✅ 特定输出格式", "JSON、代码风格等"),
        ("✅ 特定语言或方言", "方言、专业术语"),
        ("✅ 行为调整", "更友好、更专业等"),
        ("❌ 通用知识", "用 RAG 更好"),
        ("❌ 实时信息", "用工具调用"),
        ("❌ 简单任务", "Prompt Engineering 足够")
    ]

    for use_case, desc in use_cases:
        console.print(f"{use_case}: {desc}")

    console.print("\n[yellow]2. 训练技巧[/yellow]\n")

    tips = [
        "💡 从小数据集开始，验证流程",
        "💡 使用较小的模型快速迭代",
        "💡 监控训练和验证 loss",
        "💡 保存多个 checkpoint",
        "💡 定期测试生成质量",
        "💡 使用学习率调度器",
        "💡 梯度累积处理大 batch"
    ]

    for tip in tips:
        console.print(tip)

    console.print("\n[yellow]3. 常见问题[/yellow]\n")

    issues = [
        {
            "problem": "Loss 不下降",
            "causes": ["学习率太小", "数据问题", "模型配置错误"],
            "solutions": ["增大学习率", "检查数据", "检查配置"]
        },
        {
            "problem": "过拟合",
            "causes": ["数据太少", "训练太久", "模型太大"],
            "solutions": ["增加数据", "早停", "增加 dropout"]
        },
        {
            "problem": "显存不足",
            "causes": ["batch size 太大", "模型太大", "序列太长"],
            "solutions": ["减小 batch size", "使用量化", "减小 max_length"]
        }
    ]

    for issue in issues:
        console.print(f"[red]问题:[/red] {issue['problem']}")
        console.print(f"[yellow]原因:[/yellow] {', '.join(issue['causes'])}")
        console.print(f"[green]解决:[/green] {', '.join(issue['solutions'])}\n")


def main():
    """主函数"""
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]LoRA 微调示例[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    demos = [
        ("数据准备", demo_data_preparation),
        ("LoRA 配置", demo_lora_configuration),
        ("训练流程", demo_training_process),
        ("评估方法", demo_evaluation),
        ("部署流程", demo_deployment),
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
