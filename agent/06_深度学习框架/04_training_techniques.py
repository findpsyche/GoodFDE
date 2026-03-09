"""
深度学习训练技巧

本示例展示如何：
1. 使用学习率调度器优化训练
2. 实现梯度累积以模拟大 batch
3. 使用混合精度训练加速
4. 应用梯度裁剪防止梯度爆炸
5. 选择合适的权重初始化方法
6. 使用正则化技术防止过拟合
7. 了解分布式训练基础

学习目标：
- 掌握常用的训练优化技巧
- 理解学习率调度的原理和选择
- 能够在显存受限时使用梯度累积
- 了解混合精度训练的优势和使用方法

前置要求：
- 完成 01_pytorch_basics.py
- 理解基本的训练循环
- 了解梯度下降优化
"""

# ============================================================
# 标准库导入
# ============================================================
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# ============================================================
# 第三方库导入（带错误处理）
# ============================================================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich import print as rprint

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

except ImportError as e:
    print(f"错误: 缺少必要的库 - {e}")
    print("请安装: pip install torch numpy matplotlib rich")
    sys.exit(1)

# ============================================================
# 环境设置
# ============================================================
console = Console()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


@dataclass
class SchedulerInfo:
    """调度器信息"""
    name: str
    description: str
    use_case: str
    params: Dict[str, float]


@dataclass
class TrainingResult:
    """训练结果"""
    method: str
    final_loss: float
    training_time: float
    memory_used: float = 0.0


# ============================================================
# 简单模型定义
# ============================================================

class SimpleNet(nn.Module):
    """简单的全连接网络用于演示"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_classes: int = 2, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class DeepNet(nn.Module):
    """深层网络用于测试梯度问题"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_layers: int = 5, num_classes: int = 2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================
# 辅助函数
# ============================================================

def generate_synthetic_data(n_samples: int = 1000, input_dim: int = 10, num_classes: int = 2, noise: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成合成数据"""
    X = torch.randn(n_samples, input_dim)
    # 创建线性可分的数据 + 噪声
    weights = torch.randn(input_dim, num_classes)
    y_logits = X @ weights + noise * torch.randn(n_samples, num_classes)
    y = torch.argmax(y_logits, dim=1)
    return X, y


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                    criterion: nn.Module, device: torch.device, accumulation_steps: int = 1,
                    scaler: Optional[GradScaler] = None, clip_grad: Optional[float] = None) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        if scaler is not None:
            # 混合精度训练
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # 常规训练
            outputs = model(inputs)
            loss = criterion(outputs, targets) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    # 处理剩余的 batch
    if len(dataloader) % accumulation_steps != 0:
        if clip_grad is not None:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """评估模型"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================
# Demo 1: 学习率调度
# ============================================================

def demo_learning_rate_scheduling():
    """演示学习率调度器"""
    console.print(Panel.fit("学习率调度器演示", style="bold blue"))

    # 生成数据
    X_train, y_train = generate_synthetic_data(n_samples=1000, input_dim=10, num_classes=2)
    X_test, y_test = generate_synthetic_data(n_samples=200, input_dim=10, num_classes=2)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 定义不同的调度器
    schedulers_info = [
        SchedulerInfo(
            name="StepLR",
            description="每隔 step_size 个 epoch 将学习率乘以 gamma",
            use_case="稳定的训练，需要阶段性降低学习率",
            params={"step_size": 3, "gamma": 0.1}
        ),
        SchedulerInfo(
            name="CosineAnnealingLR",
            description="使用余弦退火调整学习率",
            use_case="希望学习率平滑下降，适合大多数场景",
            params={"T_max": 10, "eta_min": 1e-6}
        ),
        SchedulerInfo(
            name="LinearLR",
            description="线性调整学习率（warmup + decay）",
            use_case="训练初期需要 warmup 稳定训练",
            params={"start_factor": 0.1, "total_iters": 5}
        ),
        SchedulerInfo(
            name="OneCycleLR",
            description="在一个周期内先升后降学习率",
            use_case="快速训练，单周期训练策略",
            params={"max_lr": 0.01, "total_steps": 50, "pct_start": 0.3}
        ),
        SchedulerInfo(
            name="CosineAnnealingWarmRestarts",
            description="余弦退火 + 周期性重启",
            use_case="希望跳出局部最优，持续优化",
            params={"T_0": 5, "T_mult": 2}
        ),
    ]

    # 可视化学习率曲线
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('不同学习率调度器对比', fontsize=16)

    results = []

    for idx, scheduler_info in enumerate(schedulers_info):
        ax = axes[idx // 3, idx % 3]

        # 创建模型和优化器
        model = SimpleNet(input_dim=10, hidden_dim=64, num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 创建调度器
        if scheduler_info.name == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_info.params)
        elif scheduler_info.name == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_info.params)
        elif scheduler_info.name == "LinearLR":
            scheduler = optim.lr_scheduler.LinearLR(optimizer, **scheduler_info.params)
        elif scheduler_info.name == "OneCycleLR":
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_info.params)
        elif scheduler_info.name == "CosineAnnealingWarmRestarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_info.params)

        # 记录学习率变化
        lr_history = []
        loss_history = []

        # 训练
        for epoch in range(10):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_loss = evaluate(model, test_loader, criterion, device)

            lr_history.append(optimizer.param_groups[0]['lr'])
            loss_history.append(test_loss)

            scheduler.step()

        # 绘制学习率曲线
        ax.plot(lr_history, marker='o', label='学习率', color='blue')
        ax.set_title(f'{scheduler_info.name}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('学习率')
        ax.grid(True, alpha=0.3)
        ax.legend()

        results.append({
            'name': scheduler_info.name,
            'final_loss': test_loss,
            'final_lr': lr_history[-1]
        })

    # 隐藏多余的子图
    if len(schedulers_info) < 6:
        for idx in range(len(schedulers_info), 6):
            axes[idx // 3, idx % 3].set_visible(False)

    plt.tight_layout()
    plt.savefig('S:/新建文件夹/工作内容/AI岗位/06_深度学习框架/lr_schedulers_comparison.png', dpi=150, bbox_inches='tight')
    console.print("\n[green]✓[/green] 学习率曲线图已保存为 lr_schedulers_comparison.png")

    # 对比表格
    table = Table(title="\n学习率调度器对比")
    table.add_column("调度器", style="cyan")
    table.add_column("最终 Loss", style="magenta")
    table.add_column("适用场景", style="yellow")
    table.add_column("特点", style="green")

    for scheduler_info, result in zip(schedulers_info, results):
        table.add_row(
            scheduler_info.name,
            f"{result['final_loss']:.4f}",
            scheduler_info.use_case,
            scheduler_info.description
        )

    console.print(table)

    console.print("\n[bold cyan]关键要点：[/bold cyan]")
    console.print("• [yellow]StepLR[/yellow]: 简单直观，适合稳定下降")
    console.print("• [yellow]CosineAnnealingLR[/yellow]: 平滑下降，推荐默认选择")
    console.print("• [yellow]LinearLR[/yellow]: 适合 warmup，通常配合其他调度器")
    console.print("• [yellow]OneCycleLR[/yellow]: 快速训练，单周期场景")
    console.print("• [yellow]CosineAnnealingWarmRestarts[/yellow]: 周期重启，避免局部最优")


# ============================================================
# Demo 2: 梯度累积
# ============================================================

def demo_gradient_accumulation():
    """演示梯度累积"""
    console.print(Panel.fit("\n梯度累积演示", style="bold blue"))

    console.print("\n[bold cyan]梯度累积原理：[/bold cyan]")
    console.print("当显存不足以支持大 batch size 时，可以通过累积多个小 batch 的梯度")
    console.print("来模拟大 batch 的训练效果。")
    console.print("\n公式：有效 batch size = 小 batch size × 累积步数")
    console.print("例如：batch_size=8, accumulation_steps=4 → 等效于 batch_size=32")

    # 生成数据
    X_train, y_train = generate_synthetic_data(n_samples=1000, input_dim=10, num_classes=2)
    X_test, y_test = generate_synthetic_data(n_samples=200, input_dim=10, num_classes=2)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # 创建不同的 DataLoader
    train_loader_32 = DataLoader(train_dataset, batch_size=32, shuffle=True)
    train_loader_8 = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    results = []

    # 方法 1: 直接使用 batch_size=32
    console.print("\n[bold yellow]方法 1: batch_size=32（基准）[/bold yellow]")
    model1 = SimpleNet(input_dim=10, hidden_dim=64, num_classes=2).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    criterion1 = nn.CrossEntropyLoss()

    start_time = time.time()
    losses1 = []
    for epoch in range(5):
        loss = train_one_epoch(model1, train_loader_32, optimizer1, criterion1, device)
        test_loss = evaluate(model1, test_loader, criterion1, device)
        losses1.append(test_loss)
    time1 = time.time() - start_time

    console.print(f"最终测试 Loss: {test_loss:.4f}")
    console.print(f"训练时间: {time1:.2f} 秒")
    results.append(TrainingResult(
        method="batch_size=32",
        final_loss=test_loss,
        training_time=time1
    ))

    # 方法 2: batch_size=8 + accumulation_steps=4
    console.print("\n[bold yellow]方法 2: batch_size=8 + accumulation_steps=4（梯度累积）[/bold yellow]")
    model2 = SimpleNet(input_dim=10, hidden_dim=64, num_classes=2).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    criterion2 = nn.CrossEntropyLoss()

    start_time = time.time()
    losses2 = []
    for epoch in range(5):
        loss = train_one_epoch(model2, train_loader_8, optimizer2, criterion2, device, accumulation_steps=4)
        test_loss = evaluate(model2, test_loader, criterion2, device)
        losses2.append(test_loss)
    time2 = time.time() - start_time

    console.print(f"最终测试 Loss: {test_loss:.4f}")
    console.print(f"训练时间: {time2:.2f} 秒")
    results.append(TrainingResult(
        method="batch_size=8 + acc=4",
        final_loss=test_loss,
        training_time=time2
    ))

    # 方法 3: batch_size=8（无累积，作为对比）
    console.print("\n[bold yellow]方法 3: batch_size=8（无累积，小 batch）[/bold yellow]")
    model3 = SimpleNet(input_dim=10, hidden_dim=64, num_classes=2).to(device)
    optimizer3 = optim.Adam(model3.parameters(), lr=0.001)
    criterion3 = nn.CrossEntropyLoss()

    start_time = time.time()
    losses3 = []
    for epoch in range(5):
        loss = train_one_epoch(model3, train_loader_8, optimizer3, criterion3, device)
        test_loss = evaluate(model3, test_loader, criterion3, device)
        losses3.append(test_loss)
    time3 = time.time() - start_time

    console.print(f"最终测试 Loss: {test_loss:.4f}")
    console.print(f"训练时间: {time3:.2f} 秒")
    results.append(TrainingResult(
        method="batch_size=8",
        final_loss=test_loss,
        training_time=time3
    ))

    # 结果对比
    table = Table(title="\n梯度累积效果对比")
    table.add_column("方法", style="cyan")
    table.add_column("等效 Batch", style="yellow")
    table.add_column("最终 Loss", style="magenta")
    table.add_column("训练时间 (秒)", style="green")

    table.add_row(
        "直接大 batch",
        "32",
        f"{results[0].final_loss:.4f}",
        f"{results[0].training_time:.2f}"
    )
    table.add_row(
        "梯度累积",
        "8 × 4 = 32",
        f"{results[1].final_loss:.4f}",
        f"{results[1].training_time:.2f}"
    )
    table.add_row(
        "小 batch",
        "8",
        f"{results[2].final_loss:.4f}",
        f"{results[2].training_time:.2f}"
    )

    console.print(table)

    # 绘制 loss 对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses1, marker='o', label='batch_size=32', linewidth=2)
    ax.plot(losses2, marker='s', label='batch_size=8 + acc=4', linewidth=2)
    ax.plot(losses3, marker='^', label='batch_size=8', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('测试 Loss', fontsize=12)
    ax.set_title('梯度累积效果对比', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('S:/新建文件夹/工作内容/AI岗位/06_深度学习框架/gradient_accumulation_comparison.png', dpi=150, bbox_inches='tight')
    console.print("\n[green]✓[/green] 对比图已保存为 gradient_accumulation_comparison.png")

    console.print("\n[bold cyan]关键要点：[/bold cyan]")
    console.print("• 梯度累积可以达到和大 batch 相似的效果")
    console.print("• 优势：在显存受限时仍然可以使用大 batch 的训练策略")
    console.print("• 注意：需要调整学习率（通常按累积步数的比例缩放）")
    console.print("• Batch Normalization 在小 batch + 累积时表现可能不同")


# ============================================================
# Demo 3: 混合精度训练
# ============================================================

def demo_mixed_precision():
    """演示混合精度训练"""
    console.print(Panel.fit("\n混合精度训练演示", style="bold blue"))

    console.print("\n[bold cyan]精度类型说明：[/bold cyan]")
    console.print("• [yellow]FP32 (float32)[/yellow]: 32 位浮点数，标准精度")
    console.print("• [yellow]FP16 (float16)[/yellow]: 16 位浮点数，半精度，显存减半，速度更快")
    console.print("• [yellow]BF16 (bfloat16)[/yellow]: 16 位脑浮点数，动态范围与 FP32 相同")
    console.print("\n[bold cyan]混合精度训练：[/bold cyan]")
    console.print("部分计算使用 FP16 加速，关键部分保持 FP32 保证精度")
    console.print("使用 Loss Scaling 防止梯度下溢")

    # 生成数据
    X_train, y_train = generate_synthetic_data(n_samples=2000, input_dim=100, num_classes=10)
    X_test, y_test = generate_synthetic_data(n_samples=400, input_dim=100, num_classes=10)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    results = []

    # 全精度训练
    console.print("\n[bold yellow]全精度训练 (FP32)[/bold yellow]")
    model_fp32 = SimpleNet(input_dim=100, hidden_dim=128, num_classes=10).to(device)
    optimizer_fp32 = optim.Adam(model_fp32.parameters(), lr=0.001)
    criterion_fp32 = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    losses_fp32 = []
    for epoch in range(10):
        loss = train_one_epoch(model_fp32, train_loader, optimizer_fp32, criterion_fp32, device)
        test_loss = evaluate(model_fp32, test_loader, criterion_fp32, device)
        losses_fp32.append(test_loss)
    time_fp32 = time.time() - start_time

    if device.type == 'cuda':
        memory_fp32 = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    else:
        memory_fp32 = 0.0

    console.print(f"最终测试 Loss: {test_loss:.4f}")
    console.print(f"训练时间: {time_fp32:.2f} 秒")
    if device.type == 'cuda':
        console.print(f"峰值显存: {memory_fp32:.2f} MB")

    results.append(TrainingResult(
        method="FP32",
        final_loss=test_loss,
        training_time=time_fp32,
        memory_used=memory_fp32
    ))

    # 混合精度训练
    if device.type == 'cuda':
        console.print("\n[bold yellow]混合精度训练 (FP16 + FP32)[/bold yellow]")
        model_amp = SimpleNet(input_dim=100, hidden_dim=128, num_classes=10).to(device)
        optimizer_amp = optim.Adam(model_amp.parameters(), lr=0.001)
        criterion_amp = nn.CrossEntropyLoss()
        scaler = GradScaler()

        torch.cuda.reset_peak_memory_stats(device)

        start_time = time.time()
        losses_amp = []
        for epoch in range(10):
            loss = train_one_epoch(model_amp, train_loader, optimizer_amp, criterion_amp, device, scaler=scaler)
            test_loss = evaluate(model_amp, test_loader, criterion_amp, device)
            losses_amp.append(test_loss)
        time_amp = time.time() - start_time

        memory_amp = torch.cuda.max_memory_allocated(device) / 1024**2  # MB

        console.print(f"最终测试 Loss: {test_loss:.4f}")
        console.print(f"训练时间: {time_amp:.2f} 秒")
        console.print(f"峰值显存: {memory_amp:.2f} MB")

        results.append(TrainingResult(
            method="混合精度",
            final_loss=test_loss,
            training_time=time_amp,
            memory_used=memory_amp
        ))

        # 结果对比
        table = Table(title="\n混合精度 vs 全精度对比")
        table.add_column("精度模式", style="cyan")
        table.add_column("最终 Loss", style="magenta")
        table.add_column("训练时间 (秒)", style="yellow")
        table.add_column("峰值显存 (MB)", style="green")
        table.add_column("加速比", style="blue")

        speedup = time_fp32 / time_amp
        memory_reduction = (1 - memory_amp / memory_fp32) * 100

        table.add_row(
            "FP32",
            f"{results[0].final_loss:.4f}",
            f"{results[0].training_time:.2f}",
            f"{results[0].memory_used:.2f}",
            "-"
        )
        table.add_row(
            "混合精度 (AMP)",
            f"{results[1].final_loss:.4f}",
            f"{results[1].training_time:.2f}",
            f"{results[1].memory_used:.2f}",
            f"{speedup:.2f}x"
        )

        console.print(table)

        # 绘制对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss 对比
        ax1.plot(losses_fp32, marker='o', label='FP32', linewidth=2)
        ax1.plot(losses_amp, marker='s', label='混合精度', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('测试 Loss', fontsize=12)
        ax1.set_title('训练 Loss 对比', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 资源对比
        metrics = ['训练时间\n(秒)', '峰值显存\n(MB)']
        fp32_values = [time_fp32, memory_fp32]
        amp_values = [time_amp, memory_amp]

        x = np.arange(len(metrics))
        width = 0.35

        ax2.bar(x - width/2, fp32_values, width, label='FP32', alpha=0.8)
        ax2.bar(x + width/2, amp_values, width, label='混合精度', alpha=0.8)
        ax2.set_ylabel('数值', fontsize=12)
        ax2.set_title('资源使用对比', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('S:/新建文件夹/工作内容/AI岗位/06_深度学习框架/mixed_precision_comparison.png', dpi=150, bbox_inches='tight')
        console.print("\n[green]✓[/green] 对比图已保存为 mixed_precision_comparison.png")

        console.print(f"\n[bold cyan]关键要点：[/bold cyan]")
        console.print(f"• [yellow]速度提升[/yellow]: {speedup:.2f}x 加速")
        console.print(f"• [yellow]显存节省[/yellow]: {memory_reduction:.1f}% 显存减少")
        console.print(f"• [yellow]精度保持[/yellow]: Loss 基本一致，精度无损")
        console.print("• 混合精度训练在大模型上效果更明显")
        console.print("• 注意：某些操作不支持 FP16，会自动回退到 FP32")
    else:
        console.print("\n[yellow]⚠[/yellow] 混合精度训练需要 CUDA 支持，跳过对比")


# ============================================================
# Demo 4: 梯度裁剪
# ============================================================

def demo_gradient_clipping():
    """演示梯度裁剪"""
    console.print(Panel.fit("\n梯度裁剪演示", style="bold blue"))

    console.print("\n[bold cyan]梯度爆炸问题：[/bold cyan]")
    console.print("在深层网络或 RNN 中，梯度可能变得非常大，导致：")
    console.print("• 权重更新过大，模型参数爆炸")
    console.print("• 训练不稳定，Loss 变成 NaN")
    console.print("• 模型无法收敛")

    # 生成容易发生梯度爆炸的数据
    X_train, y_train = generate_synthetic_data(n_samples=500, input_dim=10, num_classes=2, noise=0.5)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 使用深层网络更容易出现梯度问题
    results = []

    # 无梯度裁剪
    console.print("\n[bold yellow]无梯度裁剪训练[/bold yellow]")
    model_no_clip = DeepNet(input_dim=10, hidden_dim=64, num_layers=8, num_classes=2).to(device)
    optimizer_no_clip = optim.SGD(model_no_clip.parameters(), lr=0.1)  # 大学习率更容易出问题
    criterion = nn.CrossEntropyLoss()

    grad_norms_no_clip = []
    losses_no_clip = []

    for epoch in range(10):
        model_no_clip.train()
        total_loss = 0.0
        total_grad_norm = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer_no_clip.zero_grad()
            outputs = model_no_clip(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # 计算梯度范数
            grad_norm = torch.nn.utils.clip_grad_norm_(model_no_clip.parameters(), float('inf'))
            total_grad_norm += grad_norm

            optimizer_no_clip.step()
            total_loss += loss.item()

        avg_grad_norm = total_grad_norm / len(train_loader)
        avg_loss = total_loss / len(train_loader)

        grad_norms_no_clip.append(avg_grad_norm)
        losses_no_clip.append(avg_loss)

    console.print(f"最终 Loss: {avg_loss:.4f}")
    console.print(f"最大梯度范数: {max(grad_norms_no_clip):.4f}")

    results.append({
        'method': '无裁剪',
        'final_loss': avg_loss,
        'max_grad_norm': max(grad_norms_no_clip)
    })

    # 使用梯度裁剪
    console.print("\n[bold yellow]梯度裁剪训练 (clip_grad_norm_)[/bold yellow]")
    model_clip = DeepNet(input_dim=10, hidden_dim=64, num_layers=8, num_classes=2).to(device)
    optimizer_clip = optim.SGD(model_clip.parameters(), lr=0.1)

    grad_norms_clip = []
    losses_clip = []
    max_norm = 1.0  # 裁剪阈值

    for epoch in range(10):
        model_clip.train()
        total_loss = 0.0
        total_grad_norm = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer_clip.zero_grad()
            outputs = model_clip(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # 计算裁剪前的梯度范数
            grad_norm_before = torch.nn.utils.clip_grad_norm_(model_clip.parameters(), float('inf'))

            # 应用梯度裁剪
            torch.nn.utils.clip_grad_norm_(model_clip.parameters(), max_norm)

            total_grad_norm += grad_norm_before
            optimizer_clip.step()
            total_loss += loss.item()

        avg_grad_norm = total_grad_norm / len(train_loader)
        avg_loss = total_loss / len(train_loader)

        grad_norms_clip.append(avg_grad_norm)
        losses_clip.append(avg_loss)

    console.print(f"最终 Loss: {avg_loss:.4f}")
    console.print(f"最大梯度范数（裁剪前）: {max(grad_norms_clip):.4f}")
    console.print(f"裁剪阈值: {max_norm}")

    results.append({
        'method': f'梯度裁剪 ({max_norm})',
        'final_loss': avg_loss,
        'max_grad_norm': max(grad_norms_clip)
    })

    # 结果表格
    table = Table(title="\n梯度裁剪效果对比")
    table.add_column("方法", style="cyan")
    table.add_column("最终 Loss", style="magenta")
    table.add_column("最大梯度范数", style="yellow")

    for result in results:
        table.add_row(
            result['method'],
            f"{result['final_loss']:.4f}",
            f"{result['max_grad_norm']:.4f}"
        )

    console.print(table)

    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 梯度范数对比
    ax1.plot(grad_norms_no_clip, marker='o', label='无裁剪', linewidth=2, color='red')
    ax1.axhline(y=max_norm, color='blue', linestyle='--', label=f'裁剪阈值 ({max_norm})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('梯度范数', fontsize=12)
    ax1.set_title('梯度范数变化', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Loss 对比
    ax2.plot(losses_no_clip, marker='o', label='无裁剪', linewidth=2, color='red')
    ax2.plot(losses_clip, marker='s', label='梯度裁剪', linewidth=2, color='blue')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('训练 Loss', fontsize=12)
    ax2.set_title('训练 Loss 对比', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('S:/新建文件夹/工作内容/AI岗位/06_深度学习框架/gradient_clipping_comparison.png', dpi=150, bbox_inches='tight')
    console.print("\n[green]✓[/green] 对比图已保存为 gradient_clipping_comparison.png")

    console.print("\n[bold cyan]关键要点：[/bold cyan]")
    console.print("• [yellow]clip_grad_norm_[/yellow]: 按范数裁剪，保持梯度方向")
    console.print("• [yellow]clip_grad_value_[/yellow]: 按值裁剪，限制每个梯度元素")
    console.print("• 典型阈值: 0.5 - 5.0，根据任务调整")
    console.print("• RNN、Transformer 等模型中梯度裁剪很重要")


# ============================================================
# Demo 5: 权重初始化
# ============================================================

def demo_weight_initialization():
    """演示权重初始化"""
    console.print(Panel.fit("\n权重初始化演示", style="bold blue"))

    console.print("\n[bold cyan]权重初始化的重要性：[/bold cyan]")
    console.print("良好的初始化可以：")
    console.print("• 加速训练收敛")
    console.print("• 避免梯度消失/爆炸")
    console.print("• 提高最终模型性能")

    # 定义不同的初始化方法
    def init_xavier(model):
        """Xavier/Glorot 初始化 - 适合 tanh, sigmoid"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def init_kaiming(model):
        """Kaiming/He 初始化 - 适合 ReLU"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def init_normal(model, mean=0, std=0.01):
        """正态分布初始化"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=mean, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def init_uniform(model, a=0, b=0.1):
        """均匀分布初始化"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, a=a, b=b)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # 生成数据
    X_train, y_train = generate_synthetic_data(n_samples=1000, input_dim=10, num_classes=2)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    init_methods = [
        ('Xavier Uniform', init_xavier, '适合 tanh, sigmoid 激活函数'),
        ('Kaiming Normal', init_kaiming, '适合 ReLU 激活函数'),
        ('Normal (0, 0.01)', lambda m: init_normal(m, 0, 0.01), '小随机值，保守初始化'),
        ('Uniform (0, 0.1)', lambda m: init_uniform(m, 0, 0.1), '均匀分布初始化'),
    ]

    results = []

    # 绘制初始化分布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('不同初始化方法的权重分布', fontsize=16)

    for idx, (name, init_func, description) in enumerate(init_methods):
        ax = axes[idx // 2, idx % 2]

        # 创建模型并应用初始化
        model = SimpleNet(input_dim=10, hidden_dim=64, num_classes=2)
        init_func(model)

        # 收集第一层权重
        fc1_weights = model.fc1.weight.detach().numpy().flatten()

        # 绘制分布
        ax.hist(fc1_weights, bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'{name}\n{description}', fontsize=10)
        ax.set_xlabel('权重值', fontsize=9)
        ax.set_ylabel('频数', fontsize=9)
        ax.grid(True, alpha=0.3)

        # 训练并记录结果
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for epoch in range(10):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            losses.append(loss)

        results.append({
            'name': name,
            'final_loss': loss,
            'weight_mean': fc1_weights.mean(),
            'weight_std': fc1_weights.std(),
            'weight_min': fc1_weights.min(),
            'weight_max': fc1_weights.max()
        })

    plt.tight_layout()
    plt.savefig('S:/新建文件夹/工作内容/AI岗位/06_深度学习框架/weight_initialization_distribution.png', dpi=150, bbox_inches='tight')
    console.print("\n[green]✓[/green] 权重分布图已保存为 weight_initialization_distribution.png")

    # 训练曲线对比
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (name, init_func, description) in enumerate(init_methods):
        model = SimpleNet(input_dim=10, hidden_dim=64, num_classes=2).to(device)
        init_func(model)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for epoch in range(10):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            losses.append(loss)

        ax.plot(losses, marker='o', label=name, linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('训练 Loss', fontsize=12)
    ax.set_title('不同初始化方法的训练曲线', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('S:/新建文件夹/工作内容/AI岗位/06_深度学习框架/weight_initialization_training.png', dpi=150, bbox_inches='tight')
    console.print("[green]✓[/green] 训练曲线图已保存为 weight_initialization_training.png")

    # 结果表格
    table = Table(title="\n权重初始化对比")
    table.add_column("初始化方法", style="cyan")
    table.add_column("最终 Loss", style="magenta")
    table.add_column("权重均值", style="yellow")
    table.add_column("权重标准差", style="green")

    for result in results:
        table.add_row(
            result['name'],
            f"{result['final_loss']:.4f}",
            f"{result['weight_mean']:.4f}",
            f"{result['weight_std']:.4f}"
        )

    console.print(table)

    console.print("\n[bold cyan]关键要点：[/bold cyan]")
    console.print("• [yellow]Xavier 初始化[/yellow]: 适合 tanh, sigmoid，保持方差一致")
    console.print("• [yellow]Kaiming 初始化[/yellow]: 适合 ReLU 系列，考虑 ReLU 的特性")
    console.print("• [yellow]小随机值[/yellow]: 适合浅层网络，深层网络可能梯度消失")
    console.print("• PyTorch 默认初始化通常已经很好，只在特殊需求时自定义")


# ============================================================
# Demo 6: 正则化技术
# ============================================================

def demo_regularization():
    """演示正则化技术"""
    console.print(Panel.fit("\n正则化技术演示", style="bold blue"))

    console.print("\n[bold cyan]正则化技术：[/bold cyan]")
    console.print("正则化用于防止过拟合，提高模型泛化能力：")
    console.print("• [yellow]Dropout[/yellow]: 随机丢弃神经元，防止共适应")
    console.print("• [yellow]Weight Decay[/yellow]: L2 正则化，惩罚大权重")
    console.print("• [yellow]Label Smoothing[/yellow]: 标签平滑，防止过度自信")

    # 生成容易过拟合的小数据集
    X_train, y_train = generate_synthetic_data(n_samples=200, input_dim=20, num_classes=3, noise=0.3)
    X_test, y_test = generate_synthetic_data(n_samples=200, input_dim=20, num_classes=3, noise=0.3)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 使用大容量模型容易过拟合
    class BigNet(nn.Module):
        def __init__(self, dropout=0.0):
            super().__init__()
            self.fc1 = nn.Linear(20, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 128)
            self.fc4 = nn.Linear(128, 3)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.relu(self.fc3(x))
            x = self.dropout(x)
            x = self.fc4(x)
            return x

    results = []

    # 1. 无正则化
    console.print("\n[bold yellow]无正则化（基准）[/bold yellow]")
    model_no_reg = BigNet(dropout=0.0).to(device)
    optimizer_no_reg = optim.Adam(model_no_reg.parameters(), lr=0.01, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    train_losses_no_reg = []
    test_losses_no_reg = []

    for epoch in range(50):
        train_loss = train_one_epoch(model_no_reg, train_loader, optimizer_no_reg, criterion, device)
        test_loss = evaluate(model_no_reg, test_loader, criterion, device)
        train_losses_no_reg.append(train_loss)
        test_losses_no_reg.append(test_loss)

    console.print(f"最终训练 Loss: {train_loss:.4f}")
    console.print(f"最终测试 Loss: {test_loss:.4f}")
    console.print(f"过拟合程度: {test_loss - train_loss:.4f}")

    results.append({
        'method': '无正则化',
        'train_loss': train_loss,
        'test_loss': test_loss,
        'overfitting': test_loss - train_loss
    })

    # 2. Dropout
    console.print("\n[bold yellow]使用 Dropout (p=0.5)[/bold yellow]")
    model_dropout = BigNet(dropout=0.5).to(device)
    optimizer_dropout = optim.Adam(model_dropout.parameters(), lr=0.01, weight_decay=0.0)

    train_losses_dropout = []
    test_losses_dropout = []

    for epoch in range(50):
        train_loss = train_one_epoch(model_dropout, train_loader, optimizer_dropout, criterion, device)
        test_loss = evaluate(model_dropout, test_loader, criterion, device)
        train_losses_dropout.append(train_loss)
        test_losses_dropout.append(test_loss)

    console.print(f"最终训练 Loss: {train_loss:.4f}")
    console.print(f"最终测试 Loss: {test_loss:.4f}")
    console.print(f"过拟合程度: {test_loss - train_loss:.4f}")

    results.append({
        'method': 'Dropout',
        'train_loss': train_loss,
        'test_loss': test_loss,
        'overfitting': test_loss - train_loss
    })

    # 3. Weight Decay
    console.print("\n[bold yellow]使用 Weight Decay (L2正则, 1e-4)[/bold yellow]")
    model_wd = BigNet(dropout=0.0).to(device)
    optimizer_wd = optim.Adam(model_wd.parameters(), lr=0.01, weight_decay=1e-4)

    train_losses_wd = []
    test_losses_wd = []

    for epoch in range(50):
        train_loss = train_one_epoch(model_wd, train_loader, optimizer_wd, criterion, device)
        test_loss = evaluate(model_wd, test_loader, criterion, device)
        train_losses_wd.append(train_loss)
        test_losses_wd.append(test_loss)

    console.print(f"最终训练 Loss: {train_loss:.4f}")
    console.print(f"最终测试 Loss: {test_loss:.4f}")
    console.print(f"过拟合程度: {test_loss - train_loss:.4f}")

    results.append({
        'method': 'Weight Decay',
        'train_loss': train_loss,
        'test_loss': test_loss,
        'overfitting': test_loss - train_loss
    })

    # 4. Dropout + Weight Decay
    console.print("\n[bold yellow]使用 Dropout + Weight Decay[/bold yellow]")
    model_combined = BigNet(dropout=0.5).to(device)
    optimizer_combined = optim.Adam(model_combined.parameters(), lr=0.01, weight_decay=1e-4)

    train_losses_combined = []
    test_losses_combined = []

    for epoch in range(50):
        train_loss = train_one_epoch(model_combined, train_loader, optimizer_combined, criterion, device)
        test_loss = evaluate(model_combined, test_loader, criterion, device)
        train_losses_combined.append(train_loss)
        test_losses_combined.append(test_loss)

    console.print(f"最终训练 Loss: {train_loss:.4f}")
    console.print(f"最终测试 Loss: {test_loss:.4f}")
    console.print(f"过拟合程度: {test_loss - train_loss:.4f}")

    results.append({
        'method': 'Dropout + WD',
        'train_loss': train_loss,
        'test_loss': test_loss,
        'overfitting': test_loss - train_loss
    })

    # 结果表格
    table = Table(title="\n正则化技术效果对比")
    table.add_column("方法", style="cyan")
    table.add_column("训练 Loss", style="yellow")
    table.add_column("测试 Loss", style="magenta")
    table.add_column("过拟合程度", style="red")

    for result in results:
        overfit_color = "green" if result['overfitting'] < 0.5 else "red"
        table.add_row(
            result['method'],
            f"{result['train_loss']:.4f}",
            f"{result['test_loss']:.4f}",
            f"[{overfit_color}]{result['overfitting']:.4f}[/{overfit_color}]"
        )

    console.print(table)

    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 训练 Loss 对比
    ax1.plot(train_losses_no_reg, label='无正则化', linewidth=2)
    ax1.plot(train_losses_dropout, label='Dropout', linewidth=2)
    ax1.plot(train_losses_wd, label='Weight Decay', linewidth=2)
    ax1.plot(train_losses_combined, label='Dropout + WD', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('训练 Loss', fontsize=12)
    ax1.set_title('训练 Loss 对比', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 测试 Loss 对比
    ax2.plot(test_losses_no_reg, label='无正则化', linewidth=2)
    ax2.plot(test_losses_dropout, label='Dropout', linewidth=2)
    ax2.plot(test_losses_wd, label='Weight Decay', linewidth=2)
    ax2.plot(test_losses_combined, label='Dropout + WD', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('测试 Loss', fontsize=12)
    ax2.set_title('测试 Loss 对比（越低越好）', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('S:/新建文件夹/工作内容/AI岗位/06_深度学习框架/regularization_comparison.png', dpi=150, bbox_inches='tight')
    console.print("\n[green]✓[/green] 对比图已保存为 regularization_comparison.png")

    console.print("\n[bold cyan]关键要点：[/bold cyan]")
    console.print("• [yellow]Dropout[/yellow]: 训练时随机丢弃，测试时使用全部权重，效果好")
    console.print("• [yellow]Weight Decay[/yellow]: 简单有效，适合大多数情况")
    console.print("• [yellow]组合使用[/yellow]: Dropout + Weight Decay 通常效果最好")
    console.print("• [yellow]Label Smoothing[/yellow]: 适合分类任务，防止过度自信")
    console.print("• 正则化强度需要根据任务和数据量调整")


# ============================================================
# Demo 7: 分布式训练基础
# ============================================================

def demo_distributed_basics():
    """演示分布式训练基础"""
    console.print(Panel.fit("\n分布式训练基础演示", style="bold blue"))

    console.print("\n[bold cyan]分布式训练模式：[/bold cyan]")

    table = Table(title="\nDataParallel vs DistributedDataParallel")
    table.add_column("特性", style="cyan")
    table.add_column("DataParallel (DP)", style="yellow")
    table.add_column("DistributedDataParallel (DDP)", style="green")

    table.add_row(
        "使用场景",
        "单机多卡，原型开发",
        "单机多卡 / 多机多卡，生产环境"
    )
    table.add_row(
        "实现方式",
        "Python 线程并行",
        "多进程并行（每个 GPU 一个进程）"
    )
    table.add_row(
        "性能",
        "较低（GIL 限制）",
        "高（真正并行）"
    )
    table.add_row(
        "代码复杂度",
        "简单（一行代码）",
        "中等（需要初始化进程组）"
    )
    table.add_row(
        "推荐用途",
        "快速实验，小模型",
        "大规模训练，生产环境"
    )

    console.print(table)

    # 生成数据
    X_train, y_train = generate_synthetic_data(n_samples=500, input_dim=10, num_classes=2)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 创建一个简单模型
    model = SimpleNet(input_dim=10, hidden_dim=64, num_classes=2)

    # 演示 DataParallel 的使用
    if torch.cuda.device_count() > 1:
        console.print(f"\n[green]检测到 {torch.cuda.device_count()} 个 GPU[/green]")

        console.print("\n[bold yellow]DataParallel 使用示例：[/bold yellow]")
        console.print("```python")
        console.print("# 将模型包装为 DataParallel")
        console.print("model = nn.DataParallel(model)")
        console.print("model = model.to(device)")
        console.print("")
        console.print("# 正常训练，DataParallel 会自动：")
        console.print("# 1. 将输入数据分割到各个 GPU")
        console.print("# 2. 在各 GPU 上并行前向计算")
        console.print("# 3. 收集梯度并更新模型")
        console.print("```")

        # 实际演示
        console.print("\n[bold cyan]实际演示：[/bold cyan]")
        model_dp = SimpleNet(input_dim=10, hidden_dim=64, num_classes=2)
        model_dp = nn.DataParallel(model_dp)
        model_dp = model_dp.to(device)

        console.print(f"模型设备: {next(model_dp.parameters()).device}")
        console.print(f"GPU 数量: {model_dp.device_ids if hasattr(model_dp, 'device_ids') else 'N/A'}")

        # 训练一个 epoch
        optimizer = optim.Adam(model_dp.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()
        model_dp.train()
        total_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model_dp(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - start_time

        console.print(f"训练完成，平均 Loss: {avg_loss:.4f}")
        console.print(f"训练时间: {elapsed:.2f} 秒")

    else:
        console.print("\n[yellow]⚠[/yellow] 检测到单 GPU 或 CPU，无法演示多 GPU 并行")
        console.print("\n[bold yellow]DataParallel 使用示例：[/bold yellow]")
        console.print("```python")
        console.print("# 创建模型")
        console.print("model = SimpleNet().to(device)")
        console.print("")
        console.print("# 如果有多个 GPU，包装为 DataParallel")
        console.print("if torch.cuda.device_count() > 1:")
        console.print("    model = nn.DataParallel(model)")
        console.print("")
        console.print("# 之后正常训练")
        console.print("for data, target in dataloader:")
        console.print("    output = model(data)")
        console.print("    loss = criterion(output, target)")
        console.print("    loss.backward()")
        console.print("    optimizer.step()")
        console.print("```")

    # DDP 代码示例
    console.print("\n[bold yellow]DistributedDataParallel (DDP) 使用示例：[/bold yellow]")
    console.print("```python")
    console.print("import torch.distributed as dist")
    console.print("from torch.nn.parallel import DistributedDataParallel as DDP")
    console.print("")
    console.print("# 初始化进程组")
    console.print("dist.init_process_group('nccl')  # 或 'gloo' for CPU")
    console.print("")
    console.print("# 创建模型并移至当前 GPU")
    console.print("model = SimpleNet().to(local_rank)")
    console.print("")
    console.print("# 包装为 DDP")
    console.print("model = DDP(model, device_ids=[local_rank])")
    console.print("")
    console.print("# 使用 DistributedSampler")
    console.print("sampler = DistributedSampler(dataset)")
    console.print("dataloader = DataLoader(dataset, sampler=sampler, ...)")
    console.print("")
    console.print("# 训练时设置 epoch")
    console.print("for epoch in range(epochs):")
    console.print("    sampler.set_epoch(epoch)")
    console.print("    for data, target in dataloader:")
    console.print("        # 正常训练代码")
    console.print("```")

    console.print("\n[bold cyan]关键要点：[/bold cyan]")
    console.print("• [yellow]DataParallel[/yellow]: 简单但性能一般，适合快速实验")
    console.print("• [yellow]DistributedDataParallel[/yellow]: 性能更好，适合生产环境")
    console.print("• DDP 需要多进程，使用 torchrun 或 python -m torch.distributed.launch 启动")
    console.print("• DDP 每个进程独立，需要使用 DistributedSampler 分割数据")
    console.print("• 对于大规模训练，优先选择 DDP")

    console.print("\n[bold cyan]启动 DDP 训练命令示例：[/bold cyan]")
    console.print("```bash")
    console.print("# 单机 4 GPU")
    console.print("torchrun --nproc_per_node=4 train.py")
    console.print("")
    console.print("# 多机训练（2 个节点，每节点 4 GPU）")
    console.print("# 节点 0（主节点）:")
    console.print("torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \\")
    console.print("        --master_addr='主节点IP' --master_port=29500 train.py")
    console.print("")
    console.print("# 节点 1:")
    console.print("torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \\")
    console.print("        --master_addr='主节点IP' --master_port=29500 train.py")
    console.print("```")


# ============================================================
# 主函数
# ============================================================

def print_menu():
    """打印菜单"""
    menu_table = Table(title="\n深度学习训练技巧演示", show_header=True, header_style="bold magenta")
    menu_table.add_column("序号", style="cyan", width=6)
    menu_table.add_column("演示内容", style="yellow")
    menu_table.add_column("说明", style="green")

    menu_table.add_row("1", "学习率调度", "对比多种学习率调度器")
    menu_table.add_row("2", "梯度累积", "显存受限时模拟大 batch")
    menu_table.add_row("3", "混合精度训练", "加速训练并节省显存")
    menu_table.add_row("4", "梯度裁剪", "防止梯度爆炸")
    menu_table.add_row("5", "权重初始化", "对比不同初始化方法")
    menu_table.add_row("6", "正则化技术", "防止过拟合")
    menu_table.add_row("7", "分布式训练基础", "多 GPU 并行训练")
    menu_table.add_row("0", "退出程序", "")

    console.print(menu_table)


def main():
    """主函数"""
    console.print(Panel.fit("深度学习训练技巧", style="bold cyan"))
    console.print(f"\n当前设备: {device}")
    if device.type == 'cuda':
        console.print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
        console.print(f"GPU 数量: {torch.cuda.device_count()}")

    demos = {
        '1': ('学习率调度', demo_learning_rate_scheduling),
        '2': ('梯度累积', demo_gradient_accumulation),
        '3': ('混合精度训练', demo_mixed_precision),
        '4': ('梯度裁剪', demo_gradient_clipping),
        '5': ('权重初始化', demo_weight_initialization),
        '6': ('正则化技术', demo_regularization),
        '7': ('分布式训练基础', demo_distributed_basics),
    }

    while True:
        print_menu()
        choice = console.input("\n[bold cyan]请选择演示内容 (输入序号或 0 退出): [/bold cyan]").strip()

        if choice == '0':
            console.print("\n[green]感谢使用！再见！[/green]")
            break

        if choice in demos:
            name, demo_func = demos[choice]
            console.print(f"\n[bold cyan]正在运行: {name}[/bold cyan]")
            console.print("=" * 50)
            try:
                demo_func()
            except Exception as e:
                console.print(f"\n[red]错误: {e}[/red]")
                import traceback
                traceback.print_exc()
            console.print("\n" + "=" * 50)
        else:
            console.print("\n[red]无效的选择，请重新输入！[/red]")


if __name__ == "__main__":
    main()
