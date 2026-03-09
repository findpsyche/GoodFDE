"""
PyTorch 基础教程

本示例展示如何：
1. 创建和操作 Tensor
2. 使用自动微分计算梯度
3. 构建神经网络模块
4. 实现完整的训练循环
5. 使用 DataLoader 加载数据
6. 管理计算设备（CPU/GPU）

学习目标：
- 掌握 PyTorch Tensor 的基本操作
- 理解自动微分机制
- 能够定义和训练简单的神经网络
- 熟悉 PyTorch 的数据加载流程

前置要求：
- Python 3.10+
- PyTorch 2.1+
- 基础的线性代数知识
"""

import os
import time
import json
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

load_dotenv()
console = Console()


@dataclass
class TrainingMetrics:
    """训练指标数据类"""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    learning_rate: float


class SimpleDataset(Dataset):
    """简单的自定义数据集"""

    def __init__(self, num_samples: int = 1000, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        # 生成随机数据
        self.data = torch.randn(num_samples, 10)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


class SimpleClassifier(nn.Module):
    """简单的分类器模型"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def demo_tensor_basics():
    """演示 Tensor 基础操作"""
    console.print("\n[bold cyan]=== Tensor 基础操作 ===[/bold cyan]")

    # 1. 创建 Tensor 的各种方式
    console.print("\n[bold]1. 创建 Tensor[/bold]")

    console.print("\n[yellow]从列表创建:[/yellow]")
    t_list = torch.tensor([1, 2, 3, 4])
    console.print(f"  torch.tensor([1, 2, 3, 4]) = {t_list}")

    console.print("\n[yellow]创建全零 Tensor:[/yellow]")
    t_zeros = torch.zeros(2, 3)
    console.print(f"  torch.zeros(2, 3) = \n{t_zeros}")

    console.print("\n[yellow]创建全一 Tensor:[/yellow]")
    t_ones = torch.ones(2, 3)
    console.print(f"  torch.ones(2, 3) = \n{t_ones}")

    console.print("\n[yellow]创建单位矩阵:[/yellow]")
    t_eye = torch.eye(3)
    console.print(f"  torch.eye(3) = \n{t_eye}")

    console.print("\n[yellow]创建随机 Tensor (正态分布):[/yellow]")
    t_randn = torch.randn(2, 3)
    console.print(f"  torch.randn(2, 3) = \n{t_randn}")

    console.print("\n[yellow]创建随机 Tensor (均匀分布):[/yellow]")
    t_rand = torch.rand(2, 3)
    console.print(f"  torch.rand(2, 3) = \n{t_rand}")

    console.print("\n[yellow]创建等差数列:[/yellow]")
    t_arange = torch.arange(0, 10, 2)
    console.print(f"  torch.arange(0, 10, 2) = {t_arange}")

    # 2. Tensor 属性
    console.print("\n[bold]2. Tensor 属性[/bold]")
    t = torch.randn(3, 4)
    console.print(f"\n  Tensor: \n{t}")
    console.print(f"  形状 (shape): {t.shape}")
    console.print(f"  数据类型 (dtype): {t.dtype}")
    console.print(f"  设备 (device): {t.device}")
    console.print(f"  元素数量: {t.numel()}")

    # 3. 基本运算
    console.print("\n[bold]3. 基本运算[/bold]")

    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

    console.print(f"\n[yellow]Tensor A:[/yellow]\n{a}")
    console.print(f"[yellow]Tensor B:[/yellow]\n{b}")

    console.print("\n[yellow]加法:[/yellow]")
    console.print(f"  A + B = \n{a + b}")

    console.print("\n[yellow]减法:[/yellow]")
    console.print(f"  A - B = \n{a - b}")

    console.print("\n[yellow]逐元素乘法:[/yellow]")
    console.print(f"  A * B = \n{a * b}")

    console.print("\n[yellow]矩阵乘法:[/yellow]")
    console.print(f"  A @ B = \n{a @ b}")
    console.print(f"  torch.matmul(A, B) = \n{torch.matmul(a, b)}")

    console.print("\n[yellow]广播机制:[/yellow]")
    c = torch.tensor([[1, 2, 3], [4, 5, 6]])
    d = torch.tensor([10, 20, 30])
    console.print(f"  C (shape={c.shape}):\n{c}")
    console.print(f"  D (shape={d.shape}): {d}")
    console.print(f"  C + D = \n{c + d}")

    # 4. 索引和切片
    console.print("\n[bold]4. 索引和切片[/bold]")

    t = torch.arange(12).reshape(3, 4)
    console.print(f"\n[yellow]原始 Tensor:[/yellow]\n{t}")

    console.print(f"\n[yellow]获取第 0 行:[/yellow] {t[0]}")
    console.print(f"[yellow]获取第 1 列:[/yellow] {t[:, 1]}")
    console.print(f"[yellow]获取第 0-1 行，第 1-2 列:[/yellow]\n{t[0:2, 1:3]}")
    console.print(f"[yellow]使用掩码索引:[/yellow] {t[t > 5]}")

    # 5. 形状操作
    console.print("\n[bold]5. 形状操作[/bold]")

    t = torch.arange(12)
    console.print(f"\n[yellow]原始 Tensor (shape={t.shape}):[/yellow] {t}")

    console.print(f"\n[yellow]reshape 为 (3, 4):[/yellow]")
    t_reshaped = t.reshape(3, 4)
    console.print(f"{t_reshaped}")

    console.print(f"\n[yellow]view 为 (4, 3):[/yellow]")
    t_viewed = t.view(4, 3)
    console.print(f"{t_viewed}")

    console.print(f"\n[yellow]增加维度 (unsqueeze):[/yellow]")
    t_unsqueezed = t.unsqueeze(0)  # 在第 0 维增加
    console.print(f"  unsqueeze(0): shape {t.shape} -> {t_unsqueezed.shape}")

    console.print(f"\n[yellow]减少维度 (squeeze):[/yellow]")
    t_squeezed = t_unsqueezed.squeeze()
    console.print(f"  squeeze(): shape {t_unsqueezed.shape} -> {t_squeezed.shape}")

    console.print(f"\n[yellow]转置:[/yellow]")
    t_t = t_reshaped.T
    console.print(f"  (3, 4) -> {t_t.shape}")
    console.print(f"{t_t}")

    # 6. 设备转移
    console.print("\n[bold]6. 设备转移[/bold]")

    t_cpu = torch.randn(2, 3)
    console.print(f"\n[yellow]CPU Tensor:[/yellow] device={t_cpu.device}")

    if torch.cuda.is_available():
        t_cuda = t_cpu.to('cuda')
        console.print(f"[yellow]GPU Tensor:[/yellow] device={t_cuda.device}")
        t_back = t_cuda.to('cpu')
        console.print(f"[yellow]转回 CPU:[/yellow] device={t_back.device}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        t_mps = t_cpu.to('mps')
        console.print(f"[yellow]MPS (Apple Silicon) Tensor:[/yellow] device={t_mps.device}")
    else:
        console.print("[yellow]没有可用的 GPU，跳过 GPU 演示[/yellow]")

    # 7. 与 NumPy 互转
    if NUMPY_AVAILABLE:
        console.print("\n[bold]7. 与 NumPy 互转[/bold]")

        console.print("\n[yellow]Tensor -> NumPy:[/yellow]")
        t = torch.tensor([1, 2, 3])
        np_array = t.numpy()
        console.print(f"  Tensor: {t}")
        console.print(f"  NumPy: {np_array}")

        console.print("\n[yellow]NumPy -> Tensor:[/yellow]")
        np_array = np.array([4, 5, 6])
        t = torch.from_numpy(np_array)
        console.print(f"  NumPy: {np_array}")
        console.print(f"  Tensor: {t}")
    else:
        console.print("\n[bold]7. NumPy 不可用，跳过互转演示[/bold]")

    console.print("\n[green]✓ Tensor 基础操作演示完成[/green]")


def demo_autograd():
    """演示自动微分"""
    console.print("\n[bold cyan]=== 自动微分 ===[/bold cyan]")

    # 1. 基本梯度计算
    console.print("\n[bold]1. 基本梯度计算[/bold]")

    console.print("\n[yellow]创建需要梯度的 Tensor:[/yellow]")
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    console.print(f"  x = {x}, requires_grad={x.requires_grad}")

    console.print("\n[yellow]定义计算: y = x^2 + 2x + 1[/yellow]")
    y = x ** 2 + 2 * x + 1
    console.print(f"  y = {y}")

    console.print("\n[yellow]反向传播计算梯度:[/yellow]")
    y.backward(torch.ones_like(y))
    console.print(f"  dy/dx = {x.grad}")

    console.print("\n[dim]理论验证: dy/dx = 2x + 2[/dim]")
    console.print(f"[dim]  x=[2, 3] 时，梯度应为 [6, 8][/dim]")

    # 2. 计算图可视化
    console.print("\n[bold]2. 计算图示例[/bold]")

    console.print("\n[yellow]构建计算图: z = (a + b) * (b - 1)[/yellow]")
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)

    c = a + b
    d = b - 1
    z = c * d

    console.print(f"  a = {a}, b = {b}")
    console.print(f"  c = a + b = {c}")
    console.print(f"  d = b - 1 = {d}")
    console.print(f"  z = c * d = {z}")

    z.backward()
    console.print(f"\n[yellow]反向传播:[/yellow]")
    console.print(f"  dz/da = {a.grad} (应为 (b-1) = {b.item() - 1})")
    console.print(f"  dz/db = {b.grad} (应为 (a+b) + (b-1) = {a.item() + b.item() + b.item() - 1})")

    # 3. 梯度累积和清零
    console.print("\n[bold]3. 梯度累积[/bold]")

    x = torch.tensor(1.0, requires_grad=True)
    console.print(f"\n[yellow]初始 x = {x}[/yellow]")

    for i in range(3):
        y = x ** 2
        y.backward()
        console.print(f"  第 {i+1} 次反向传播后: grad = {x.grad}")

    console.print("\n[yellow]梯度会累积！需要手动清零:[/yellow]")
    x.grad.zero_()
    console.print(f"  清零后: grad = {x.grad}")

    # 4. detach() 和 no_grad()
    console.print("\n[bold]4. 停止梯度追踪[/bold]")

    x = torch.tensor(2.0, requires_grad=True)
    console.print(f"\n[yellow]x = {x}, requires_grad={x.requires_grad}[/yellow]")

    console.print("\n[yellow]使用 detach() 创建不需要梯度的副本:[/yellow]")
    x_detached = x.detach()
    console.print(f"  x_detached.requires_grad = {x_detached.requires_grad}")

    console.print("\n[yellow]使用 torch.no_grad() 上下文:[/yellow]")
    with torch.no_grad():
        y = x ** 2
        console.print(f"  在 no_grad() 中计算 y = x^2")
        console.print(f"  y.requires_grad = {y.requires_grad}")

    # 5. 自定义函数梯度验证
    console.print("\n[bold]5. 数值梯度 vs 自动梯度验证[/bold]")

    def f(x: float) -> float:
        """测试函数: f(x) = x^3"""
        return x ** 3

    def numerical_grad(f, x: float, h: float = 1e-5) -> float:
        """数值梯度计算"""
        return (f(x + h) - f(x - h)) / (2 * h)

    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 3
    y.backward()
    auto_grad = x.grad.item()

    num_grad = numerical_grad(lambda t: t ** 3, 2.0)

    console.print(f"\n[yellow]函数: f(x) = x^3, 在 x=2 处[/yellow]")
    console.print(f"  数值梯度: {num_grad:.6f}")
    console.print(f"  自动梯度: {auto_grad:.6f}")
    console.print(f"  理论梯度: {3 * 2 ** 2:.6f} (dy/dx = 3x^2)")
    console.print(f"  差异: {abs(auto_grad - num_grad):.8f}")

    console.print("\n[green]✓ 自动微分明示完成[/green]")


def demo_nn_module():
    """演示神经网络模块"""
    console.print("\n[bold cyan]=== 神经网络模块 ===[/bold cyan]")

    # 1. 定义简单的线性模型
    console.print("\n[bold]1. 定义简单的线性模型[/bold]")

    class LinearModel(nn.Module):
        """简单的线性模型: y = wx + b"""

        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    model = LinearModel(10, 2)
    console.print(f"\n[yellow]模型结构:[/yellow]")
    console.print(model)

    # 2. 前向传播
    console.print("\n[bold]2. 前向传播[/bold]")

    x = torch.randn(5, 10)  # batch_size=5, input_dim=10
    output = model(x)
    console.print(f"\n[yellow]输入 shape:[/yellow] {x.shape}")
    console.print(f"[yellow]输出 shape:[/yellow] {output.shape}")
    console.print(f"[yellow]输出值:[/yellow]\n{output}")

    # 3. 常用层
    console.print("\n[bold]3. 常用神经网络层[/bold]")

    console.print("\n[yellow]Linear 层:[/yellow]")
    linear = nn.Linear(10, 5)
    console.print(f"  nn.Linear(10, 5)")
    console.print(f"  权重 shape: {linear.weight.shape}")
    console.print(f"  偏置 shape: {linear.bias.shape}")

    console.print("\n[yellow]Conv2d 层:[/yellow]")
    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    console.print(f"  nn.Conv2d(3, 16, kernel_size=3, padding=1)")
    x_img = torch.randn(1, 3, 32, 32)
    y_conv = conv(x_img)
    console.print(f"  输入: {x_img.shape} -> 输出: {y_conv.shape}")

    console.print("\n[yellow]BatchNorm1d 层:[/yellow]")
    bn = nn.BatchNorm1d(64)
    console.print(f"  nn.BatchNorm1d(64)")
    console.print(f"  可学习参数数量: {sum(p.numel() for p in bn.parameters())}")

    console.print("\n[yellow]Dropout 层:[/yellow]")
    dropout = nn.Dropout(p=0.5)
    console.print(f"  nn.Dropout(p=0.5)")
    x_train = torch.ones(10)
    x_drop = dropout(x_train)
    console.print(f"  训练模式: 输入全1 -> 输出有{x_drop.eq(0).sum().item()}个0")

    dropout.eval()  # 切换到评估模式
    x_eval = dropout(x_train)
    console.print(f"  评估模式: 输入全1 -> 输出全1 (无dropout)")

    # 4. 查看模型参数
    console.print("\n[bold]4. 查看模型参数[/bold]")

    console.print("\n[yellow]所有参数:[/yellow]")
    for name, param in model.named_parameters():
        console.print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

    console.print("\n[yellow]参数总数:[/yellow]")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"  总参数: {total_params}")
    console.print(f"  可训练参数: {trainable_params}")

    # 5. 模型保存和加载
    console.print("\n[bold]5. 模型保存和加载[/bold]")

    # 保存 state_dict
    state_dict = model.state_dict()
    console.print(f"\n[yellow]state_dict 包含的参数:[/yellow]")
    for key in state_dict.keys():
        console.print(f"  {key}")

    # 创建新模型并加载
    new_model = LinearModel(10, 2)
    new_model.load_state_dict(state_dict)
    console.print("\n[yellow]成功加载参数到新模型[/yellow]")

    # 6. 定义 MLP 分类器
    console.print("\n[bold]6. 完整的 MLP 分类器[/bold]")

    mlp = SimpleClassifier(input_dim=10, hidden_dim=64, num_classes=2)
    console.print(f"\n[yellow]MLP 结构:[/yellow]")
    console.print(mlp)

    console.print(f"\n[yellow]测试前向传播:[/yellow]")
    x = torch.randn(8, 10)  # batch_size=8
    output = mlp(x)
    console.print(f"  输入 shape: {x.shape}")
    console.print(f"  输出 shape: {output.shape}")
    console.print(f"  预测类别: {output.argmax(dim=1)}")

    console.print("\n[green]✓ 神经网络模块演示完成[/green]")


def demo_training_loop():
    """演示完整训练循环"""
    console.print("\n[bold cyan]=== 完整训练循环 ===[/bold cyan]")

    # 1. 生成合成数据
    console.print("\n[bold]1. 生成合成数据[/bold]")

    torch.manual_seed(42)
    num_samples = 1000
    input_dim = 20

    console.print(f"\n[yellow]生成 {num_samples} 个样本，{input_dim} 维特征[/yellow]")

    # 生成数据
    X = torch.randn(num_samples, input_dim)
    # 创建简单的线性决策边界
    true_weights = torch.randn(input_dim, 1)
    y_logits = X @ true_weights + torch.randn(num_samples, 1) * 0.1
    y = (y_logits > 0).long().squeeze()

    console.print(f"  正样本: {y.sum().item()}, 负样本: {(1 - y).sum().item()}")

    # 划分训练集和验证集
    split = int(0.8 * num_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    console.print(f"  训练集: {X_train.shape[0]} 样本")
    console.print(f"  验证集: {X_val.shape[0]} 样本")

    # 2. 定义模型、损失函数、优化器
    console.print("\n[bold]2. 定义模型、损失函数、优化器[/bold]")

    model = SimpleClassifier(input_dim=input_dim, hidden_dim=64, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    console.print(f"\n[yellow]模型:[/yellow]")
    console.print(model)
    console.print(f"\n[yellow]损失函数: CrossEntropyLoss[/yellow]")
    console.print(f"[yellow]优化器: Adam (lr=0.001)[/yellow]")

    # 3. 训练循环
    console.print("\n[bold]3. 训练模型[/bold]")

    num_epochs = 10
    batch_size = 32
    history: List[TrainingMetrics] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("训练中...", total=num_epochs)

        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            # 小批量训练
            indices = torch.randperm(X_train.shape[0])
            for i in range(0, X_train.shape[0], batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_x = X_train[batch_indices]
                batch_y = y_train[batch_indices]

                # 前向传播
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_x.shape[0]
                _, predicted = outputs.max(1)
                train_total += batch_y.shape[0]
                train_correct += predicted.eq(batch_y).sum().item()

            train_loss /= train_total
            train_acc = train_correct / train_total

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for i in range(0, X_val.shape[0], batch_size):
                    batch_x = X_val[i:i + batch_size]
                    batch_y = y_val[i:i + batch_size]

                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item() * batch_x.shape[0]
                    _, predicted = outputs.max(1)
                    val_total += batch_y.shape[0]
                    val_correct += predicted.eq(batch_y).sum().item()

            val_loss /= val_total
            val_acc = val_correct / val_total

            # 记录指标
            history.append(TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                learning_rate=optimizer.param_groups[0]['lr']
            ))

            progress.update(task, advance=1)

    console.print("\n[green]✓ 训练完成[/green]")

    # 4. 展示训练结果
    console.print("\n[bold]4. 训练结果[/bold]")

    table = Table(title="训练历史")
    table.add_column("Epoch", style="cyan")
    table.add_column("Train Loss", style="red")
    table.add_column("Train Acc", style="green")
    table.add_column("Val Loss", style="red")
    table.add_column("Val Acc", style="green")

    for metrics in history[-5:]:  # 只显示最后5个epoch
        table.add_row(
            str(metrics.epoch),
            f"{metrics.train_loss:.4f}",
            f"{metrics.train_acc:.4f}",
            f"{metrics.val_loss:.4f}",
            f"{metrics.val_acc:.4f}"
        )

    console.print(table)

    # 5. 绘制 loss 曲线
    if MATPLOTLIB_AVAILABLE:
        console.print("\n[bold]5. 生成训练曲线[/bold]")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss 曲线
        epochs = [m.epoch for m in history]
        train_losses = [m.train_loss for m in history]
        val_losses = [m.val_loss for m in history]

        ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
        ax1.plot(epochs, val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy 曲线
        train_accs = [m.train_acc for m in history]
        val_accs = [m.val_acc for m in history]

        ax2.plot(epochs, train_accs, label='Train Acc', marker='o')
        ax2.plot(epochs, val_accs, label='Val Acc', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # 保存图片
        output_path = "S:\\新建文件夹\\工作内容\\AI岗位\\06_深度学习框架\\training_curves.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ 训练曲线已保存到: {output_path}[/green]")
        plt.close()

    # 6. 模型评估
    console.print("\n[bold]6. 最终模型评估[/bold]")

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        _, val_predictions = val_outputs.max(1)
        final_acc = val_predictions.eq(y_val).sum().item() / y_val.shape[0]

    console.print(f"\n[yellow]验证集准确率: {final_acc:.4f} ({final_acc*100:.2f}%)[/yellow]")

    console.print("\n[green]✓ 完整训练循环演示完成[/green]")


def demo_data_loading():
    """演示数据加载"""
    console.print("\n[bold cyan]=== 数据加载 ===[/bold cyan]")

    # 1. 自定义 Dataset
    console.print("\n[bold]1. 自定义 Dataset[/bold]")

    dataset = SimpleDataset(num_samples=100)
    console.print(f"\n[yellow]数据集大小: {len(dataset)}[/yellow]")

    # 获取单个样本
    sample, label = dataset[0]
    console.print(f"[yellow]第一个样本:[/yellow]")
    console.print(f"  数据 shape: {sample.shape}")
    console.print(f"  标签: {label}")

    # 2. 数据变换
    console.print("\n[bold]2. 数据变换[/bold]")

    def normalize(x: torch.Tensor) -> torch.Tensor:
        """标准化变换"""
        return (x - x.mean()) / (x.std() + 1e-8)

    def add_noise(x: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
        """添加噪声"""
        return x + torch.randn_like(x) * noise_level

    console.print("\n[yellow]应用变换:[/yellow]")
    original_sample = dataset[0][0]
    normalized = normalize(original_sample)
    noisy = add_noise(original_sample)

    console.print(f"  原始: mean={original_sample.mean():.4f}, std={original_sample.std():.4f}")
    console.print(f"  标准化后: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
    console.print(f"  加噪声后: mean={noisy.mean():.4f}, std={noisy.std():.4f}")

    # 3. DataLoader
    console.print("\n[bold]3. DataLoader[/bold]")

    # 创建带变换的数据集
    transformed_dataset = SimpleDataset(
        num_samples=100,
        transform=lambda x: normalize(x)
    )

    # 创建 DataLoader
    dataloader = DataLoader(
        transformed_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    console.print(f"\n[yellow]DataLoader 配置:[/yellow]")
    console.print(f"  batch_size: 16")
    console.print(f"  shuffle: True")
    console.print(f"  num_workers: 0")
    console.print(f"  总批次数: {len(dataloader)}")

    # 4. 展示一个 batch
    console.print("\n[bold]4. 批量数据展示[/bold]")

    console.print("\n[yellow]遍历 DataLoader:[/yellow]")
    for i, (batch_x, batch_y) in enumerate(dataloader):
        if i >= 2:  # 只展示前2个batch
            break
        console.print(f"\n  Batch {i + 1}:")
        console.print(f"    数据 shape: {batch_x.shape}")
        console.print(f"    标签 shape: {batch_y.shape}")
        console.print(f"    标签值: {batch_y.tolist()}")

    # 5. TensorDataset 示例
    console.print("\n[bold]5. TensorDataset 快速创建数据集[/bold]")

    # 创建特征和标签
    features = torch.randn(50, 5)
    labels = torch.randint(0, 3, (50,))

    tensor_dataset = TensorDataset(features, labels)
    tensor_loader = DataLoader(tensor_dataset, batch_size=10, shuffle=True)

    console.print(f"\n[yellow]TensorDataset 大小: {len(tensor_dataset)}[/yellow]")
    console.print(f"[yellow]Batch 数量: {len(tensor_loader)}[/yellow]")

    # 展示一个batch
    batch_features, batch_labels = next(iter(tensor_loader))
    console.print(f"\n[yellow]第一个 batch:[/yellow]")
    console.print(f"  特征 shape: {batch_features.shape}")
    console.print(f"  标签 shape: {batch_labels.shape}")

    # 6. 数据增强示例
    console.print("\n[bold]6. 数据增强示例[/bold]")

    class AugmentedDataset(Dataset):
        """带数据增强的数据集"""

        def __init__(self, num_samples=100):
            self.num_samples = num_samples
            self.data = torch.randn(num_samples, 10)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            sample = self.data[idx]

            # 随机应用增强
            if torch.rand(1).item() > 0.5:
                # 添加噪声
                sample = sample + torch.randn_like(sample) * 0.1

            if torch.rand(1).item() > 0.5:
                # 随机缩放
                scale = torch.rand(1).item() * 0.5 + 0.75  # [0.75, 1.25]
                sample = sample * scale

            return sample, idx

    aug_dataset = AugmentedDataset(50)
    aug_loader = DataLoader(aug_dataset, batch_size=8, shuffle=True)

    console.print(f"\n[yellow]数据增强数据集:[/yellow]")
    console.print(f"  数据集大小: {len(aug_dataset)}")

    # 多次获取同一个样本，展示增强效果
    console.print(f"\n[yellow]同一索引的多次采样（展示随机性）:[/yellow]")
    original = aug_dataset[0][0]
    aug1 = aug_dataset[0][0]
    aug2 = aug_dataset[0][0]

    console.print(f"  原始样本均值: {original.mean():.4f}")
    console.print(f"  增强样本1均值: {aug1.mean():.4f}")
    console.print(f"  增强样本2均值: {aug2.mean():.4f}")

    console.print("\n[green]✓ 数据加载演示完成[/green]")


def demo_device_management():
    """演示设备管理"""
    console.print("\n[bold cyan]=== 设备管理 ===[/bold cyan]")

    # 1. 检测可用设备
    console.print("\n[bold]1. 检测可用设备[/bold]")

    console.print(f"\n[yellow]CPU:[/yellow] 始终可用")
    console.print(f"  设备名称: cpu")

    if torch.cuda.is_available():
        console.print(f"\n[yellow]CUDA (GPU):[/yellow] 可用")
        console.print(f"  GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            console.print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            console.print(f"    内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        console.print(f"\n[yellow]CUDA (GPU):[/yellow] 不可用")

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        console.print(f"\n[yellow]MPS (Apple Silicon):[/yellow] 可用")
        console.print(f"  适用于 Apple Silicon (M1/M2/M3) GPU")
    else:
        console.print(f"\n[yellow]MPS (Apple Silicon):[/yellow] 不可用")

    # 2. 选择最佳设备
    console.print("\n[bold]2. 自动选择最佳设备[/bold]")

    def get_device() -> torch.device:
        """获取最佳可用设备"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    device = get_device()
    console.print(f"\n[yellow]选择设备: {device}[/yellow]")

    # 3. 模型和数据转移到设备
    console.print("\n[bold]3. 转移模型和数据到设备[/bold]")

    # 创建模型
    model = SimpleClassifier(input_dim=10, hidden_dim=32, num_classes=2)
    console.print(f"\n[yellow]初始模型设备: {next(model.parameters()).device}[/yellow]")

    # 转移模型
    model = model.to(device)
    console.print(f"[yellow]转移后模型设备: {next(model.parameters()).device}[/yellow]")

    # 转移数据
    x = torch.randn(8, 10)
    y = torch.randint(0, 2, (8,))

    console.print(f"\n[yellow]初始数据设备: {x.device}[/yellow]")
    x, y = x.to(device), y.to(device)
    console.print(f"[yellow]转移后数据设备: {x.device}[/yellow]")

    # 前向传播
    with torch.no_grad():
        output = model(x)
    console.print(f"[yellow]输出设备: {output.device}[/yellow]")

    # 4. 常见设备错误
    console.print("\n[bold]4. 常见设备不匹配错误[/bold]")

    console.print("\n[yellow]错误示例:[/yellow]")
    console.print("[dim]# CPU 上的模型处理 GPU 上的数据[/dim]")
    console.print("[dim]model_cpu = model.to('cpu')[/dim]")
    console.print("[dim]x_gpu = x.to('cuda' if torch.cuda.is_available() else 'cpu')[/dim]")
    console.print("[dim]# RuntimeError: Expected all tensors to be on the same device[/dim]")

    console.print("\n[yellow]正确做法:[/yellow]")
    console.print("[dim]# 确保模型和数据在同一设备上[/dim]")
    console.print("[dim]device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')[/dim]")
    console.print("[dim]model = model.to(device)[/dim]")
    console.print("[dim]x = x.to(device)[/dim]")

    # 5. GPU 内存管理
    if torch.cuda.is_available():
        console.print("\n[bold]5. GPU 内存管理[/bold]")

        console.print(f"\n[yellow]当前内存使用:[/yellow]")
        console.print(f"  已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        console.print(f"  已缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        # 创建大张量
        large_tensor = torch.randn(1000, 1000, device='cuda')
        console.print(f"\n[yellow]创建大张量后:[/yellow]")
        console.print(f"  已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # 清空缓存
        del large_tensor
        torch.cuda.empty_cache()
        console.print(f"\n[yellow]清理后:[/yellow]")
        console.print(f"  已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        console.print(f"  已缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # 6. 设备无关代码模板
    console.print("\n[bold]6. 设备无关代码模板[/bold]")

    console.print("\n[yellow]推荐的训练循环模式:[/yellow]")
    code_template = '''
def train_on_device(model, train_loader, device, epochs=10):
    """设备无关的训练函数"""
    model = model.to(device)  # 转移模型
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            # 关键：转移数据到同一设备
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 训练步骤
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    '''

    console.print(Panel(code_template, title="代码模板", border_style="cyan"))

    console.print("\n[green]✓ 设备管理演示完成[/green]")


def main():
    """主函数"""
    if not TORCH_AVAILABLE:
        console.print("[red]❌ PyTorch 未安装。请运行: pip install torch[/red]")
        return

    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  PyTorch 基础教程[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    # 显示环境信息
    console.print(f"\n[cyan]PyTorch 版本: {torch.__version__}[/cyan]")
    console.print(f"[cyan]CUDA 可用: {torch.cuda.is_available()}[/cyan]")
    if torch.cuda.is_available():
        console.print(f"[cyan]GPU: {torch.cuda.get_device_name(0)}[/cyan]")
    console.print(f"[cyan]MPS 可用: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}[/cyan]")

    demos = [
        ("Tensor 基础操作", demo_tensor_basics),
        ("自动微分", demo_autograd),
        ("神经网络模块", demo_nn_module),
        ("完整训练循环", demo_training_loop),
        ("数据加载", demo_data_loading),
        ("设备管理", demo_device_management),
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
