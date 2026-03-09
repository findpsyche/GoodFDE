"""
Hugging Face 生态系统

本示例展示如何：
1. 使用 Transformers 库进行模型推理
2. 使用 Datasets 库加载和处理数据
3. 理解不同 Tokenizer 的工作原理
4. 对比不同预训练模型的性能
5. 使用 Accelerate 和 PEFT 进行高效训练
6. 管理 Hugging Face Hub 资源

学习目标：
- 熟练使用 Pipeline API 进行快速推理
- 掌握 AutoModel/AutoTokenizer 的使用
- 理解数据集加载和预处理流程
- 了解参数高效微调（PEFT）的概念

前置要求：
- 完成 01_pytorch_basics.py
- 完成 02_transformer_architecture.py
- 网络连接（下载模型和数据集）
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# 尝试导入第三方库
try:
    import torch
    import torch.nn as nn
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint

    # Hugging Face 库（可选依赖）
    try:
        from transformers import (
            pipeline,
            AutoTokenizer,
            AutoModel,
            AutoModelForSequenceClassification,
            AutoModelForCausalLM,
            BertConfig,
        )
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        print("[警告] transformers 未安装，部分演示将跳过")
        print("安装命令: pip install transformers")

    try:
        from datasets import load_dataset, Dataset
        DATASETS_AVAILABLE = True
    except ImportError:
        DATASETS_AVAILABLE = False
        print("[警告] datasets 未安装，部分演示将跳过")
        print("安装命令: pip install datasets")

    try:
        from accelerate import Accelerator
        ACCELERATE_AVAILABLE = True
    except ImportError:
        ACCELERATE_AVAILABLE = False
        print("[警告] accelerate 未安装，部分演示将跳过")
        print("安装命令: pip install accelerate")

    try:
        from peft import LoraConfig, get_peft_model
        PEFT_AVAILABLE = True
    except ImportError:
        PEFT_AVAILABLE = False
        print("[警告] peft 未安装，部分演示将跳过")
        print("安装命令: pip install peft")

except ImportError as e:
    print(f"[错误] 缺少必需依赖: {e}")
    print("请安装: pip install torch rich")
    sys.exit(1)


# ==================== 环境设置 ====================
console = Console()

# 设置 Hugging Face 镜像（国内用户）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')


# ==================== 数据结构 ====================
@dataclass
class ModelBenchmarkResult:
    """模型基准测试结果"""
    model_name: str
    num_parameters: int
    inference_time: float
    memory_mb: float
    task: str = "text-generation"


@dataclass
class TokenizerComparison:
    """Tokenizer 对比结果"""
    tokenizer_name: str
    vocab_size: int
    max_length: int
    special_tokens: List[str]
    example_encoding: List[int]
    example_text: str


@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    num_rows: int
    num_columns: int
    features: List[str]
    splits: List[str]


# ==================== 演示函数 ====================

def demo_transformers_library() -> None:
    """演示 Transformers 库的核心功能"""
    if not TRANSFORMERS_AVAILABLE:
        console.print("[red]transformers 库未安装，跳过此演示[/red]")
        return

    console.print("\n[bold cyan]=== Transformers 库演示 ===[/bold cyan]\n")

    # 1. Pipeline API 快速推理
    console.print("[yellow]1. Pipeline API 快速推理[/yellow]")

    try:
        # 情感分析 pipeline
        console.print("\n[green]情感分析：[/green]")
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        texts = [
            "I love this product! It's amazing.",
            "This is terrible. I hate it.",
            "The weather is okay today."
        ]

        for text in texts:
            result = sentiment_pipeline(text)[0]
            label_emoji = "😊" if result['label'] == 'POSITIVE' else "😞"
            console.print(f"  {label_emoji} 文本: {text}")
            console.print(f"     情感: {result['label']} (置信度: {result['score']:.3f})")

        # 文本生成 pipeline
        console.print("\n[green]文本生成：[/green]")
        console.print("[dim]（使用小型 GPT-2 模型，首次运行会下载模型）[/dim]")

        generator = pipeline("text-generation", model="gpt2")
        prompt = "The future of AI is"
        outputs = generator(prompt, max_length=30, num_return_sequences=2)

        for i, output in enumerate(outputs, 1):
            console.print(f"  生成 {i}: {output['generated_text']}")

    except Exception as e:
        console.print(f"[red]Pipeline 演示出错: {e}[/red]")
        console.print("[yellow]可能原因: 网络连接问题或模型下载失败[/yellow]")

    # 2. AutoModel 和 AutoTokenizer
    console.print("\n[yellow]2. AutoModel 和 AutoTokenizer[/yellow]")

    try:
        model_name = "distilbert-base-uncased"

        console.print(f"\n[green]加载模型: {model_name}[/green]")

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        console.print(f"  词表大小: {tokenizer.vocab_size:,}")
        console.print(f"  最大长度: {tokenizer.model_max_length}")

        # 加载模型
        model = AutoModel.from_pretrained(model_name)

        # 计算参数量
        num_params = sum(p.numel() for p in model.parameters())
        console.print(f"  参数量: {num_params:,}")

        # 推理流程
        console.print("\n[green]完整推理流程：[/green]")

        text = "Hello, this is a test sentence."
        console.print(f"  输入文本: {text}")

        # Tokenization
        inputs = tokenizer(text, return_tensors="pt")
        console.print(f"  Token IDs: {inputs['input_ids'].flatten().tolist()}")
        console.print(f"  Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'].flatten())}")

        # Model forward
        with torch.no_grad():
            outputs = model(**inputs)

        console.print(f"  输出形状: {outputs.last_hidden_state.shape}")

        # 查看模型配置
        console.print("\n[green]模型配置：[/green]")
        config = model.config
        console.print(f"  隐藏层维度: {config.hidden_size}")
        console.print(f"  Attention heads: {config.num_attention_heads}")
        console.print(f"  Hidden layers: {config.num_hidden_layers}")

        # 保存和加载模型
        console.print("\n[green]模型保存和加载：[/green]")
        save_path = "./test_model"
        os.makedirs(save_path, exist_ok=True)

        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        console.print(f"  模型已保存到: {save_path}")

        # 从本地加载
        loaded_tokenizer = AutoTokenizer.from_pretrained(save_path)
        loaded_model = AutoModel.from_pretrained(save_path)
        console.print("  从本地加载成功 ✓")

        # 清理
        import shutil
        shutil.rmtree(save_path)
        console.print("  清理临时文件 ✓")

    except Exception as e:
        console.print(f"[red]AutoModel 演示出错: {e}[/red]")

    # 3. 不同任务类型的 Pipeline
    console.print("\n[yellow]3. 其他任务类型 Pipeline[/yellow]")

    try:
        # 问答系统
        console.print("\n[green]问答系统：[/green]")
        qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-uncased-distilled-squad"
        )

        context = """
        Hugging Face is a technology company that specializes in natural language processing.
        It created the Transformers library and hosts the Hugging Face Hub, a platform for
        sharing ML models and datasets.
        """

        questions = [
            "What does Hugging Face specialize in?",
            "What library did Hugging Face create?"
        ]

        for question in questions:
            result = qa_pipeline(question=question, context=context)
            console.print(f"  Q: {question}")
            console.print(f"  A: {result['answer']} (置信度: {result['score']:.3f})")

    except Exception as e:
        console.print(f"[red]问答演示出错: {e}[/red]")


def demo_datasets_library() -> None:
    """演示 Datasets 库的使用"""
    if not DATASETS_AVAILABLE:
        console.print("[red]datasets 库未安装，跳过此演示[/red]")
        return

    console.print("\n[bold cyan]=== Datasets 库演示 ===[/bold cyan]\n")

    try:
        # 1. 加载 Hub 上的数据集
        console.print("[yellow]1. 加载数据集[/yellow]")

        console.print("\n[green]加载 IMDB 数据集: [/green]")
        console.print("[dim]（首次运行会下载数据集）[/dim]")

        dataset = load_dataset("imdb", split="test[:1%]")  # 只加载 1% 作为演示
        console.print(f"  数据集类型: {type(dataset)}")
        console.print(f"  样本数量: {dataset.num_rows:,}")

        # 2. 查看数据集信息
        console.print("\n[yellow]2. 数据集信息[/yellow]")

        # 加载完整数据集配置
        full_dataset = load_dataset("imdb")

        info_table = Table(title="IMDB 数据集信息")
        info_table.add_column("分割", style="cyan")
        info_table.add_column("样本数", style="magenta")
        info_table.add_column("特征", style="green")

        for split_name, split_data in full_dataset.items():
            features = ", ".join(list(split_data.features.keys()))
            info_table.add_row(
                split_name,
                f"{split_data.num_rows:,}",
                features
            )

        console.print(info_table)

        # 3. 查看样本
        console.print("\n[yellow]3. 数据样本[/yellow]")

        console.print("\n[green]第一个样本：[/green]")
        first_sample = dataset[0]
        console.print(f"  标签: {first_sample['label']} ({'正面' if first_sample['label'] == 1 else '负面'})")
        console.print(f"  文本: {first_sample['text'][:200]}...")

        # 4. 数据预处理
        console.print("\n[yellow]4. 数据预处理[/yellow]")

        # 统计文本长度
        console.print("\n[green]统计文本长度：[/green]")

        def compute_length(example):
            return {"length": len(example["text"])}

        dataset = dataset.map(compute_length)

        total_length = sum(dataset["length"])
        avg_length = total_length / dataset.num_rows
        console.print(f"  平均文本长度: {avg_length:.1f} 字符")

        # Filter: 过滤短文本
        console.print("\n[green]过滤短文本（<100 字符）：[/green]")
        long_texts = dataset.filter(lambda x: x["length"] >= 100)
        console.print(f"  过滤后样本数: {long_texts.num_rows:,}")

        # Select: 选择特定样本
        console.print("\n[green]选择前 10 个样本：[/green]")
        selected = dataset.select(range(10))
        console.print(f"  选择样本数: {selected.num_rows}")

        # 5. 数据集分割
        console.print("\n[yellow]5. 数据集分割[/yellow]")

        console.print("\n[green]train_test_split: [/green]")
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        console.print(f"  训练集: {split_dataset['train'].num_rows:,}")
        console.print(f"  测试集: {split_dataset['test'].num_rows:,}")

        # 6. 创建自定义数据集
        console.print("\n[yellow]6. 自定义数据集[/yellow]")

        console.print("\n[green]从字典创建：[/green]")
        custom_data = {
            "text": ["这是正面评价", "这是负面评价", "非常棒的产品", "质量很差"],
            "label": [1, 0, 1, 0]
        }

        custom_dataset = Dataset.from_dict(custom_data)
        console.print(f"  自定义数据集大小: {custom_dataset.num_rows}")
        console.print(f"  特征: {custom_dataset.features}")

        # 展示数据
        data_table = Table(title="自定义数据集样本")
        data_table.add_column("文本", style="cyan")
        data_table.add_column("标签", style="magenta")

        for item in custom_dataset:
            label_text = "正面" if item["label"] == 1 else "负面"
            data_table.add_row(item["text"], label_text)

        console.print(data_table)

    except Exception as e:
        console.print(f"[red]Datasets 演示出错: {e}[/red]")
        console.print("[yellow]可能原因: 网络连接问题或数据集下载失败[/yellow]")


def demo_tokenizers() -> None:
    """演示不同 Tokenizer 的工作原理"""
    if not TRANSFORMERS_AVAILABLE:
        console.print("[red]transformers 库未安装，跳过此演示[/red]")
        return

    console.print("\n[bold cyan]=== Tokenizers 演示 ===[/bold cyan]\n")

    example_text = "Hello, world! This is a test."

    # 对比不同类型的 tokenizer
    tokenizer_configs = [
        ("BERT (WordPiece)", "bert-base-uncased"),
        ("DistilBERT (WordPiece)", "distilbert-base-uncased"),
        ("RoBERTa (BPE)", "roberta-base"),
        ("GPT-2 (BPE)", "gpt2"),
    ]

    comparison_results = []

    for name, model_name in tokenizer_configs:
        console.print(f"\n[yellow]{name}[/yellow]")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # 编码
            encoded = tokenizer.encode(example_text)
            decoded = tokenizer.decode(encoded)

            # 特殊 tokens
            special_tokens = []
            for token_name, token_id in tokenizer.special_tokens_map.items():
                if token_id is not None:
                    special_tokens.append(f"{token_name}: {token_id}")

            # 转换为 tokens
            tokens = tokenizer.convert_ids_to_tokens(encoded)

            console.print(f"  词表大小: {tokenizer.vocab_size:,}")
            console.print(f"  Token 数量: {len(encoded)}")
            console.print(f"  Tokens: {tokens}")
            console.print(f"  特殊 tokens: {len(special_tokens)} 个")

            comparison_results.append(TokenizerComparison(
                tokenizer_name=name,
                vocab_size=tokenizer.vocab_size,
                max_length=tokenizer.model_max_length,
                special_tokens=special_tokens,
                example_encoding=encoded,
                example_text=example_text
            ))

        except Exception as e:
            console.print(f"  [red]加载失败: {e}[/red]")

    # 对比表
    console.print("\n[yellow]Tokenizer 对比表[/yellow]")

    table = Table(title="不同 Tokenizer 的对比")
    table.add_column("模型", style="cyan")
    table.add_column("词表大小", style="magenta")
    table.add_column("Token 数量", style="green")
    table.add_column("压缩率", style="yellow")

    for result in comparison_results:
        compression_rate = len(result.example_text) / len(result.example_encoding)
        table.add_row(
            result.tokenizer_name,
            f"{result.vocab_size:,}",
            str(len(result.example_encoding)),
            f"{compression_rate:.2f}"
        )

    console.print(table)

    # Padding 和 Truncation
    console.print("\n[yellow]Padding 和 Truncation[/yellow]")

    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        texts = [
            "Short text.",
            "This is a much longer text that will be truncated if we set a max length.",
            "Medium length text here."
        ]

        console.print("\n[green]无填充/截断：[/green]")
        inputs_no_padding = tokenizer(texts)
        for i, input_ids in enumerate(inputs_no_padding["input_ids"]):
            console.print(f"  文本 {i+1}: {len(input_ids)} tokens")

        console.print("\n[green]填充到最大长度：[/green]")
        inputs_padded = tokenizer(texts, padding=True, return_tensors="pt")
        console.print(f"  填充后形状: {inputs_padded['input_ids'].shape}")

        console.print("\n[green]截断到最大长度（max_length=10）：[/green]")
        inputs_truncated = tokenizer(texts, max_length=10, truncation=True, padding=True)
        for i, input_ids in enumerate(inputs_truncated["input_ids"]):
            console.print(f"  文本 {i+1}: {len(input_ids)} tokens")

        # 展示特殊 tokens
        console.print("\n[yellow]特殊 Tokens[/yellow]")
        console.print(f"  [CLS] (序列开始): {tokenizer.cls_token_id} ({tokenizer.cls_token})")
        console.print(f"  [SEP] (序列分隔): {tokenizer.sep_token_id} ({tokenizer.sep_token})")
        console.print(f"  [PAD] (填充): {tokenizer.pad_token_id} ({tokenizer.pad_token})")
        console.print(f"  [UNK] (未知): {tokenizer.unk_token_id} ({tokenizer.unk_token})")

    except Exception as e:
        console.print(f"[red]Padding/Truncation 演示出错: {e}[/red]")


def demo_model_comparison() -> None:
    """对比不同预训练模型的性能"""
    if not TRANSFORMERS_AVAILABLE:
        console.print("[red]transformers 库未安装，跳过此演示[/red]")
        return

    console.print("\n[bold cyan]=== 模型对比演示 ===[/bold cyan]\n")

    console.print("[yellow]对比不同模型的参数量和推理速度[/yellow]")
    console.print("[dim]（首次运行会下载多个模型，需要时间）[/dim]\n")

    # 测试模型列表（从小到大）
    models_to_test = [
        ("distilbert-base-uncased", "DistilBERT Base"),
        ("bert-base-uncased", "BERT Base"),
        ("gpt2", "GPT-2 Small"),
    ]

    benchmark_results = []

    for model_name, display_name in models_to_test:
        console.print(f"[cyan]测试: {display_name}[/cyan]")

        try:
            # 加载模型
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # 计算参数量
            num_params = sum(p.numel() for p in model.parameters())

            # 创建测试输入
            text = "This is a test sentence for benchmarking models."
            inputs = tokenizer(text, return_tensors="pt")

            # 预热
            with torch.no_grad():
                _ = model(**inputs)

            # 测试推理时间
            num_runs = 10
            start_time = time.time()

            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(**inputs)

            avg_time = (time.time() - start_time) / num_runs

            # 估算内存使用
            memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

            console.print(f"  参数量: {num_params:,}")
            console.print(f"  平均推理时间: {avg_time*1000:.2f} ms")
            console.print(f"  模型大小: {memory_mb:.1f} MB\n")

            benchmark_results.append(ModelBenchmarkResult(
                model_name=display_name,
                num_parameters=num_params,
                inference_time=avg_time,
                memory_mb=memory_mb,
                task="text-encoding"
            ))

        except Exception as e:
            console.print(f"  [red]测试失败: {e}[/red]\n")

    # 展示对比结果
    if benchmark_results:
        console.print("[yellow]模型性能对比表[/yellow]\n")

        table = Table(title="预训练模型对比")
        table.add_column("模型", style="cyan")
        table.add_column("参数量", style="magenta")
        table.add_column("推理时间", style="green")
        table.add_column("模型大小", style="yellow")
        table.add_column("效率", style="blue")

        for result in benchmark_results:
            efficiency = result.num_parameters / result.inference_time
            table.add_row(
                result.model_name,
                f"{result.num_parameters:,}",
                f"{result.inference_time*1000:.2f} ms",
                f"{result.memory_mb:.1f} MB",
                f"{efficiency:,.0f}"
            )

        console.print(table)

        # 性能总结
        console.print("\n[green]性能总结：[/green]")
        fastest = min(benchmark_results, key=lambda x: x.inference_time)
        smallest = min(benchmark_results, key=lambda x: num_params)

        console.print(f"  最快: {fastest.model_name} ({fastest.inference_time*1000:.2f} ms)")
        console.print(f"  最小: {smallest.model_name} ({smallest.num_parameters:,} 参数)")


def demo_accelerate_peft() -> None:
    """演示 Accelerate 和 PEFT 的概念"""
    console.print("\n[bold cyan]=== Accelerate 和 PEFT 演示 ===[/bold cyan]\n")

    # 1. Accelerate 基础概念
    console.print("[yellow]1. Accelerate - 设备自动管理[/yellow]")

    if ACCELERATE_AVAILABLE:
        console.print("\n[green]Accelerator 基础用法：[/green]")

        accelerator = Accelerator()
        console.print(f"  当前设备: {accelerator.device}")
        console.print(f"  混合精度: {accelerator.mixed_precision}")

        # 创建一个简单的模型
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

        console.print(f"  模型设备（准备）: {next(model.parameters()).device}")

        # 使用 accelerator 准备
        model = accelerator.prepare(model)
        console.print(f"  模型设备（准备后）: {next(model.parameters()).device}")

    else:
        console.print("[dim]  accelerate 未安装，使用 PyTorch 原生方式演示[/dim]")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"  当前设备: {device}")

    # 2. PEFT 概念介绍
    console.print("\n[yellow]2. PEFT - 参数高效微调[/yellow]")

    console.print("\n[green]PEFT 方法对比：[/green]")

    peft_table = Table()
    peft_table.add_column("方法", style="cyan")
    peft_table.add_column("可训练参数", style="magenta")
    peft_table.add_column("适用场景", style="green")
    peft_table.add_column("优点", style="yellow")

    peft_info = [
        ("LoRA", "0.1%-1%", "生成式模型", "训练快，显存少"),
        ("Prefix Tuning", "<0.1%", "生成式任务", "几乎不增加参数"),
        ("Adapter", "1%-3%", "分类/生成", "模块化设计"),
        ("BitFit", "<0.1%", "快速适配", "只训练 bias"),
    ]

    for method, params, scenario, advantage in peft_info:
        peft_table.add_row(method, params, scenario, advantage)

    console.print(peft_table)

    # 3. LoRA 概念演示
    console.print("\n[yellow]3. LoRA 原理演示[/yellow]")

    console.print("\n[green]传统微调 vs LoRA: [/green]")

    # 创建一个简单的线性层演示
    original_layer = nn.Linear(100, 100)

    # 计算原始参数量
    original_params = sum(p.numel() for p in original_layer.parameters())

    console.print(f"  原始层参数量: {original_params:,}")

    # LoRA 使用低秩分解
    rank = 4  # 秩
    lora_A = nn.Linear(100, rank, bias=False)
    lora_B = nn.Linear(rank, 100, bias=False)

    # LoRA 参数量
    lora_params = sum(p.numel() for p in lora_A.parameters()) + \
                  sum(p.numel() for p in lora_B.parameters())

    console.print(f"  LoRA 参数量 (rank={rank}): {lora_params:,}")
    console.print(f"  参数减少: {(1 - lora_params / original_params) * 100:.1f}%")

    # 展示不同 rank 的参数效率
    console.print("\n[green]不同 Rank 的参数效率: [/green]")

    rank_table = Table()
    rank_table.add_column("Rank", style="cyan")
    rank_table.add_column("LoRA 参数量", style="magenta")
    rank_table.add_column("参数占比", style="green")
    rank_table.add_column("节省比例", style="yellow")

    input_dim, output_dim = 768, 768  # BERT-base 的 hidden size
    original_count = input_dim * output_dim

    for r in [1, 2, 4, 8, 16, 32]:
        lora_count = input_dim * r + r * output_dim
        ratio = lora_count / original_count * 100
        savings = 100 - ratio

        rank_table.add_row(
            str(r),
            f"{lora_count:,}",
            f"{ratio:.2f}%",
            f"{savings:.2f}%"
        )

    console.print(rank_table)

    # 4. 实际 LoRA 模型示例
    if PEFT_AVAILABLE and TRANSFORMERS_AVAILABLE:
        console.print("\n[yellow]4. PEFT 实际应用示例[/yellow]")

        try:
            console.print("\n[green]创建 LoRA 模型: [/green]")

            # 加载基础模型
            base_model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=2
            )

            base_params = sum(p.numel() for p in base_model.parameters())
            console.print(f"  基础模型参数: {base_params:,}")

            # LoRA 配置
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_lin", "v_lin"],  # DistilBERT 的 attention 层
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_CLS"
            )

            # 应用 LoRA
            lora_model = get_peft_model(base_model, lora_config)

            # 统计可训练参数
            trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in lora_model.parameters())

            console.print(f"  总参数: {total_params:,}")
            console.print(f"  可训练参数: {trainable_params:,}")
            console.print(f"  可训练参数占比: {trainable_params / total_params * 100:.2f}%")

            # 展示 LoRA 模块
            console.print("\n[green]LoRA 模块: [/green]")
            for name, module in lora_model.named_modules():
                if "lora" in name.lower():
                    console.print(f"  {name}: {module.__class__.__name__}")

        except Exception as e:
            console.print(f"[red]PEFT 示例出错: {e}[/red]")

    elif not PEFT_AVAILABLE:
        console.print("[dim]peft 未安装，跳过实际示例[/dim]")
        console.print("[yellow]安装命令: pip install peft[/yellow]")


def demo_hub_operations() -> None:
    """演示 Hugging Face Hub 操作"""
    if not TRANSFORMERS_AVAILABLE:
        console.print("[red]transformers 库未安装，跳过此演示[/red]")
        return

    console.print("\n[bold cyan]=== Hugging Face Hub 操作演示 ===[/bold cyan]\n")

    console.print("[yellow]注意: 本演示仅展示基本概念[/yellow]")
    console.print("[yellow]完整功能需要登录 HF 账号: huggingface-cli login[/yellow]\n")

    # 1. 模型搜索（模拟）
    console.print("[yellow]1. 搜索模型[/yellow]")

    popular_models = [
        ("bert-base-uncased", "BERT Base Model", "110M", "MLM"),
        ("gpt2", "GPT-2 Small", "124M", "Causal LM"),
        ("distilbert-base-uncased", "DistilBERT", "66M", "MLM"),
        ("facebook/bart-large", "BART Large", "400M", "Seq2Seq"),
    ]

    console.print("\n[green]流行的文本模型: [/green]")

    model_table = Table()
    model_table.add_column("模型 ID", style="cyan")
    model_table.add_column("描述", style="magenta")
    model_table.add_column("参数量", style="green")
    model_table.add_column("任务", style="yellow")

    for model_id, desc, params, task in popular_models:
        model_table.add_row(model_id, desc, params, task)

    console.print(model_table)

    # 2. 模型信息查看
    console.print("\n[yellow]2. 模型信息查看[/yellow]")

    try:
        model_name = "distilbert-base-uncased"
        console.print(f"\n[green]模型: {model_name}[/green]")

        # 加载配置
        config = BertConfig.from_pretrained(model_name)

        console.print(f"  架构类型: {config.architectures}")
        console.print(f"  隐藏层大小: {config.hidden_size}")
        console.print(f"  隐藏层数量: {config.num_hidden_layers}")
        console.print(f"  Attention Heads: {config.num_attention_heads}")
        console.print(f"  词汇表大小: {config.vocab_size}")
        console.print(f"  最大位置嵌入: {config.max_position_embeddings}")

        # 查看 tokenizer 配置
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        console.print(f"\n  Tokenizer 配置:")
        console.print(f"    词表大小: {tokenizer.vocab_size:,}")
        console.print(f"    最大长度: {tokenizer.model_max_length:,}")
        console.print(f"    Padding token: {tokenizer.pad_token}")

    except Exception as e:
        console.print(f"[red]获取模型信息失败: {e}[/red]")

    # 3. 缓存管理
    console.print("\n[yellow]3. 缓存管理[/yellow]")

    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.name == "nt":  # Windows
        cache_dir = os.path.expanduser("~/.cache/huggingface")

    console.print(f"\n[green]默认缓存目录: [/green]")
    console.print(f"  {cache_dir}")

    if os.path.exists(cache_dir):
        # 计算缓存大小
        def get_dir_size(path):
            total = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total += os.path.getsize(filepath)
            return total

        try:
            cache_size = get_dir_size(cache_dir) / (1024 ** 3)  # GB
            console.print(f"  缓存大小: {cache_size:.2f} GB")
        except:
            console.print("  缓存大小: 无法计算")

        # 列出缓存的部分模型
        console.print(f"\n[green]缓存的模型: [/green]")

        models_cache = os.path.join(cache_dir, "hub")
        if os.path.exists(models_cache):
            model_dirs = [d for d in os.listdir(models_cache) if d.startswith("models--")]

            for model_dir in model_dirs[:5]:  # 只显示前5个
                model_name = model_dir.replace("models--", "").replace("--", "/")
                console.print(f"  - {model_name}")

            if len(model_dirs) > 5:
                console.print(f"  ... 还有 {len(model_dirs) - 5} 个模型")

    else:
        console.print("  缓存目录不存在")

    console.print("\n[green]清理缓存命令: [/green]")
    console.print("  huggingface-cli delete-cache")

    # 4. Hub 操作指南
    console.print("\n[yellow]4. 常用 Hub 命令[/yellow]")

    commands = [
        ("登录", "huggingface-cli login"),
        ("Whoami", "huggingface-cli whoami"),
        ("上传模型", "model.push_to_hub()"),
        ("上传数据集", "dataset.push_to_hub()"),
        ("创建仓库", "huggingface-cli repo create <repo_name>"),
    ]

    cmd_table = Table()
    cmd_table.add_column("操作", style="cyan")
    cmd_table.add_column("命令", style="magenta")

    for operation, command in commands:
        cmd_table.add_row(operation, command)

    console.print(cmd_table)


# ==================== 主菜单 ====================
def main() -> None:
    """主函数：提供交互式菜单"""
    console.print(Panel.fit(
        "[bold cyan]Hugging Face 生态系统[/bold cyan]\n"
        "探索 Transformers、Datasets、PEFT 等核心库",
        title="欢迎",
        border_style="bright_blue"
    ))

    demos = [
        ("Transformers 库", demo_transformers_library),
        ("Datasets 库", demo_datasets_library),
        ("Tokenizers", demo_tokenizers),
        ("模型对比", demo_model_comparison),
        ("Accelerate 和 PEFT", demo_accelerate_peft),
        ("Hub 操作", demo_hub_operations),
    ]

    while True:
        console.print("\n[bold]请选择演示:[/bold]")

        for i, (name, _) in enumerate(demos, 1):
            console.print(f"  {i}. {name}")

        console.print("  0. 退出")

        try:
            choice = input("\n请输入选项 (0-6): ").strip()

            if choice == "0":
                console.print("[green]再见！[/green]")
                break

            if choice.isdigit() and 1 <= int(choice) <= len(demos):
                _, func = demos[int(choice) - 1]
                func()
            else:
                console.print("[red]无效选项，请重试[/red]")

        except KeyboardInterrupt:
            console.print("\n\n[green]再见！[/green]")
            break
        except Exception as e:
            console.print(f"\n[red]错误: {e}[/red]")


if __name__ == "__main__":
    main()
