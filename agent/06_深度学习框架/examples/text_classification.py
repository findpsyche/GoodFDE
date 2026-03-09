"""
Hugging Face 文本分类实践

功能：
- 使用 Hugging Face Transformers 进行情感分类
- 在 IMDB 数据集上微调 DistilBERT
- 完整的训练、验证和测试流程
- TensorBoard 日志记录
- 错误样本分析

学习目标：
- 掌握 Hugging Face 生态系统的使用
- 学习预训练模型的微调
- 理解训练监控和评估
- 实践错误分析方法

验证标准：
- 验证准确率达到 90% 以上
- 训练过程稳定，无异常波动
- 成功保存最佳模型
- 完成错误样本分析
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

console = Console()


class TextClassificationTrainer:
    """文本分类训练器"""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        max_length: int = 512,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 500,
        device: str = None
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps

        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        console.print(f"[green]使用设备: {self.device}[/green]")

        # 加载 tokenizer 和模型
        console.print(f"[yellow]加载模型: {model_name}...[/yellow]")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        console.print(f"[green]模型参数量: {num_params:,}[/green]\n")

        # TensorBoard
        log_dir = Path(__file__).parent / "runs" / f"text_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=log_dir)
        console.print(f"[green]TensorBoard 日志目录: {log_dir}[/green]")
        console.print(f"[cyan]启动 TensorBoard: tensorboard --logdir={log_dir.parent}[/cyan]\n")

        # 训练状态
        self.global_step = 0
        self.best_val_accuracy = 0.0

    def prepare_data(self, dataset_name: str = "imdb", num_train_samples: int = None, num_val_samples: int = None):
        """
        准备数据集
        Args:
            dataset_name: 数据集名称
            num_train_samples: 训练样本数（None 表示使用全部）
            num_val_samples: 验证样本数（None 表示使用全部）
        """
        console.print(f"[yellow]加载数据集: {dataset_name}...[/yellow]")

        # 加载数据集
        dataset = load_dataset(dataset_name)

        # 限制样本数（用于快速实验）
        if num_train_samples:
            dataset['train'] = dataset['train'].select(range(num_train_samples))
        if num_val_samples:
            dataset['test'] = dataset['test'].select(range(num_val_samples))

        console.print(f"[green]训练集: {len(dataset['train'])} 样本[/green]")
        console.print(f"[green]测试集: {len(dataset['test'])} 样本[/green]\n")

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )

        console.print("[yellow]Tokenizing...[/yellow]")
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # 设置格式
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        # 创建 DataLoader
        self.train_loader = DataLoader(
            tokenized_datasets['train'],
            batch_size=self.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            tokenized_datasets['test'],
            batch_size=self.batch_size
        )

        console.print("[green]数据准备完成！[/green]\n")

        # 优化器和调度器
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        total_steps = len(self.train_loader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

    def train_epoch(self) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in self.train_loader:
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # 记录
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # TensorBoard 日志
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/learning_rate', self.scheduler.get_last_lr()[0], self.global_step)

        return total_loss / num_batches

    def evaluate(self) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        评估模型
        Returns:
            (loss, accuracy, predictions, labels)
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

        return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)

    def train(self):
        """完整的训练流程"""
        console.print("[bold yellow]开始训练...[/bold yellow]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]训练进度", total=self.num_epochs)

            for epoch in range(self.num_epochs):
                # 训练
                train_loss = self.train_epoch()

                # 验证
                val_loss, val_accuracy, predictions, labels = self.evaluate()

                # TensorBoard 日志
                self.writer.add_scalar('val/loss', val_loss, epoch)
                self.writer.add_scalar('val/accuracy', val_accuracy, epoch)

                # 更新进度
                progress.update(task, advance=1)

                # 打印日志
                console.print(
                    f"Epoch [{epoch+1}/{self.num_epochs}] - "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_accuracy:.4f}"
                )

                # 保存最佳模型
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy

                    save_dir = Path(__file__).parent / "checkpoints"
                    save_dir.mkdir(exist_ok=True)

                    self.model.save_pretrained(save_dir / "text_classification_best")
                    self.tokenizer.save_pretrained(save_dir / "text_classification_best")

                    console.print(f"[green]保存最佳模型 (Acc: {val_accuracy:.4f})[/green]")

        console.print(f"\n[bold green]训练完成！[/bold green]")
        console.print(f"[green]最佳验证准确率: {self.best_val_accuracy:.4f}[/green]\n")

        self.writer.close()

    def analyze_errors(self, num_samples: int = 10):
        """
        分析错误样本
        Args:
            num_samples: 显示的错误样本数
        """
        console.print("[bold yellow]错误样本分析...[/bold yellow]\n")

        self.model.eval()
        error_samples = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                # 找出错误样本
                for i in range(len(labels)):
                    if predictions[i] != labels[i]:
                        text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                        error_samples.append({
                            'text': text[:200] + '...' if len(text) > 200 else text,
                            'true_label': labels[i].item(),
                            'pred_label': predictions[i].item(),
                            'confidence': torch.softmax(logits[i], dim=0).max().item()
                        })

                if len(error_samples) >= num_samples:
                    break

        # 显示错误样本
        label_names = ['Negative', 'Positive']

        table = Table(title="错误样本分析", show_header=True, header_style="bold magenta")
        table.add_column("文本", style="cyan", width=60)
        table.add_column("真实标签", style="green")
        table.add_column("预测标签", style="red")
        table.add_column("置信度", style="yellow")

        for sample in error_samples[:num_samples]:
            table.add_row(
                sample['text'],
                label_names[sample['true_label']],
                label_names[sample['pred_label']],
                f"{sample['confidence']:.4f}"
            )

        console.print(table)

    def print_classification_report(self):
        """打印分类报告"""
        console.print("\n[bold yellow]分类报告...[/bold yellow]\n")

        _, _, predictions, labels = self.evaluate()

        # 分类报告
        report = classification_report(
            labels,
            predictions,
            target_names=['Negative', 'Positive'],
            digits=4
        )
        console.print(report)

        # 混淆矩阵
        cm = confusion_matrix(labels, predictions)
        console.print("\n[bold]混淆矩阵:[/bold]")
        console.print(f"              Predicted")
        console.print(f"              Neg    Pos")
        console.print(f"Actual  Neg   {cm[0][0]:<6} {cm[0][1]:<6}")
        console.print(f"        Pos   {cm[1][0]:<6} {cm[1][1]:<6}")


def main():
    console.print(Panel.fit(
        "[bold cyan]Hugging Face 文本分类实践[/bold cyan]\n"
        "任务：IMDB 电影评论情感分类",
        border_style="cyan"
    ))

    # 创建训练器
    trainer = TextClassificationTrainer(
        model_name="distilbert-base-uncased",
        num_labels=2,
        max_length=256,  # 减少长度以加快训练
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=3,
        warmup_steps=500
    )

    # 准备数据（使用部分数据以加快实验）
    trainer.prepare_data(
        dataset_name="imdb",
        num_train_samples=2000,  # 使用 2000 个训练样本
        num_val_samples=500      # 使用 500 个验证样本
    )

    # 训练
    trainer.train()

    # 分类报告
    trainer.print_classification_report()

    # 错误分析
    trainer.analyze_errors(num_samples=10)

    # 验证标准检查
    console.print("\n[bold cyan]验证标准检查:[/bold cyan]")
    checks = [
        ("验证准确率 ≥ 90%", trainer.best_val_accuracy >= 0.90, f"{trainer.best_val_accuracy:.2%}"),
        ("模型成功保存", (Path(__file__).parent / "checkpoints" / "text_classification_best").exists(), "✓"),
        ("TensorBoard 日志生成", (Path(__file__).parent / "runs").exists(), "✓"),
    ]

    for check_name, passed, value in checks:
        status = "[green]✓[/green]" if passed else "[red]✗[/red]"
        console.print(f"{status} {check_name}: {value}")

    console.print("\n[bold green]实验完成！[/bold green]")
    console.print("[cyan]提示: 运行 'tensorboard --logdir=examples/runs' 查看训练曲线[/cyan]")


if __name__ == "__main__":
    main()
