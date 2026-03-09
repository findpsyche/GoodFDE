"""
深度学习框架 - 训练验证与调试工具

本模块实现训练过程的监控、验证和调试功能，包括：
1. 训练监控器 - 记录损失、学习率、梯度范数等指标
2. 早停机制 - 防止过拟合
3. TensorBoard 日志记录
4. 训练问题诊断工具
5. 模型验证工具
6. 检查点管理
7. 调试工具集

核心理念：验证驱动开发，闭环调试
"""

import sys
from pathlib import Path

# 尝试导入依赖
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    import json
    from datetime import datetime
    from collections import defaultdict
    import warnings
    warnings.filterwarnings('ignore')
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请安装: pip install torch numpy rich")
    sys.exit(1)

console = Console()


class TrainingMonitor:
    """训练监控器 - 记录和分析训练指标"""

    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.history = defaultdict(list)
        self.epoch = 0

    def record(self, metrics: dict):
        """记录一个epoch的指标"""
        self.epoch += 1
        for key, value in metrics.items():
            self.history[key].append(value)

    def get_metric(self, name: str):
        """获取指标历史"""
        return self.history.get(name, [])

    def summary(self):
        """生成训练摘要"""
        table = Table(title=f"训练摘要 (Epoch {self.epoch})")
        table.add_column("指标", style="cyan")
        table.add_column("最新值", style="green")
        table.add_column("最佳值", style="yellow")
        table.add_column("平均值", style="magenta")

        for metric, values in self.history.items():
            if values:
                latest = values[-1]
                best = min(values) if 'loss' in metric else max(values)
                avg = np.mean(values)
                table.add_row(
                    metric,
                    f"{latest:.4f}",
                    f"{best:.4f}",
                    f"{avg:.4f}"
                )

        console.print(table)

    def save(self, filename="training_history.json"):
        """保存训练历史"""
        filepath = self.log_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dict(self.history), f, indent=2, ensure_ascii=False)
        console.print(f"[green]训练历史已保存到: {filepath}[/green]")

    def check_gradient_health(self, model: nn.Module):
        """检查梯度健康状况"""
        grad_norms = []
        zero_grads = 0
        nan_grads = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)

                if grad_norm == 0:
                    zero_grads += 1
                if torch.isnan(param.grad).any():
                    nan_grads += 1

        if grad_norms:
            avg_norm = np.mean(grad_norms)
            max_norm = np.max(grad_norms)

            status = "健康"
            color = "green"

            if nan_grads > 0:
                status = "异常 (NaN梯度)"
                color = "red"
            elif zero_grads > len(grad_norms) * 0.5:
                status = "警告 (过多零梯度)"
                color = "yellow"
            elif max_norm > 100:
                status = "警告 (梯度爆炸)"
                color = "yellow"
            elif avg_norm < 1e-7:
                status = "警告 (梯度消失)"
                color = "yellow"

            console.print(f"[{color}]梯度状态: {status}[/{color}]")
            console.print(f"  平均范数: {avg_norm:.6f}")
            console.print(f"  最大范数: {max_norm:.6f}")
            console.print(f"  零梯度参数: {zero_grads}/{len(grad_norms)}")

            return {
                'status': status,
                'avg_norm': avg_norm,
                'max_norm': max_norm,
                'zero_grads': zero_grads
            }

        return None


class EarlyStopping:
    """早停机制 - 防止过拟合"""

    def __init__(self, patience=7, min_delta=0, mode='min'):
        """
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善量
            mode: 'min' 或 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """检查是否应该早停"""
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                console.print(f"[yellow]早停触发! 已等待 {self.patience} 个epoch无改善[/yellow]")

        return self.early_stop


class TensorBoardLogger:
    """TensorBoard 日志记录器（简化版）"""

    def __init__(self, log_dir="runs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.step = 0

        # 尝试导入 tensorboard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
            console.print(f"[green]TensorBoard 已启用: {log_dir}[/green]")
        except ImportError:
            self.writer = None
            self.enabled = False
            console.print("[yellow]TensorBoard 未安装，日志功能禁用[/yellow]")

    def log_scalar(self, tag, value, step=None):
        """记录标量值"""
        if self.enabled:
            step = step if step is not None else self.step
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag, tag_scalar_dict, step=None):
        """记录多个标量"""
        if self.enabled:
            step = step if step is not None else self.step
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag, values, step=None):
        """记录直方图"""
        if self.enabled:
            step = step if step is not None else self.step
            self.writer.add_histogram(tag, values, step)

    def log_model_weights(self, model, step=None):
        """记录模型权重分布"""
        if self.enabled:
            step = step if step is not None else self.step
            for name, param in model.named_parameters():
                self.writer.add_histogram(f'weights/{name}', param, step)
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/{name}', param.grad, step)

    def close(self):
        """关闭日志记录器"""
        if self.enabled:
            self.writer.close()


class ModelValidator:
    """模型验证工具"""

    @staticmethod
    def validate_output_shape(model, input_shape, expected_output_shape):
        """验证输出形状"""
        model.eval()
        with torch.no_grad():
            x = torch.randn(input_shape)
            output = model(x)

            if output.shape == expected_output_shape:
                console.print(f"[green]✓ 输出形状正确: {output.shape}[/green]")
                return True
            else:
                console.print(f"[red]✗ 输出形状错误: 期望 {expected_output_shape}, 实际 {output.shape}[/red]")
                return False

    @staticmethod
    def check_parameter_updates(model, optimizer, loss_fn):
        """检查参数是否正确更新"""
        # 保存初始参数
        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        # 执行一次训练步骤
        model.train()
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        # 检查参数变化
        updated = 0
        unchanged = 0

        for name, param in model.named_parameters():
            if not torch.equal(param, initial_params[name]):
                updated += 1
            else:
                unchanged += 1

        console.print(f"参数更新检查: {updated} 个参数已更新, {unchanged} 个参数未变化")

        if unchanged > 0:
            console.print("[yellow]警告: 部分参数未更新，可能存在问题[/yellow]")
            return False
        else:
            console.print("[green]✓ 所有参数正常更新[/green]")
            return True

    @staticmethod
    def test_overfitting_capability(model, optimizer, loss_fn, num_samples=10):
        """测试模型是否能过拟合小数据集"""
        console.print("\n[cyan]测试过拟合能力...[/cyan]")

        # 创建小数据集
        x = torch.randn(num_samples, 10)
        y = torch.randn(num_samples, 1)

        model.train()
        losses = []

        for epoch in range(100):
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if epoch % 20 == 0:
                console.print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")

        # 检查损失是否显著下降
        initial_loss = losses[0]
        final_loss = losses[-1]
        reduction = (initial_loss - final_loss) / initial_loss

        if reduction > 0.9:
            console.print(f"[green]✓ 模型能够过拟合小数据集 (损失下降 {reduction*100:.1f}%)[/green]")
            return True
        else:
            console.print(f"[red]✗ 模型无法过拟合小数据集 (损失仅下降 {reduction*100:.1f}%)[/red]")
            console.print("[yellow]可能的问题: 学习率过小、模型容量不足、或存在bug[/yellow]")
            return False


class CheckpointManager:
    """检查点管理器"""

    def __init__(self, checkpoint_dir="checkpoints", max_keep=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_keep = max_keep
        self.checkpoints = []

    def save(self, model, optimizer, epoch, metrics, filename=None):
        """保存检查点"""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"

        filepath = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint, filepath)
        self.checkpoints.append(filepath)

        # 清理旧检查点
        if len(self.checkpoints) > self.max_keep:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

        console.print(f"[green]检查点已保存: {filepath}[/green]")

    def load(self, model, optimizer=None, filename=None):
        """加载检查点"""
        if filename is None:
            # 加载最新的检查点
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
            if not checkpoints:
                console.print("[red]未找到检查点文件[/red]")
                return None
            filepath = checkpoints[-1]
        else:
            filepath = self.checkpoint_dir / filename

        if not filepath.exists():
            console.print(f"[red]检查点文件不存在: {filepath}[/red]")
            return None

        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        console.print(f"[green]检查点已加载: {filepath}[/green]")
        console.print(f"  Epoch: {checkpoint['epoch']}")
        console.print(f"  Metrics: {checkpoint['metrics']}")

        return checkpoint


class DebugTools:
    """调试工具集"""

    @staticmethod
    def print_model_summary(model, input_shape=(1, 10)):
        """打印模型摘要"""
        table = Table(title="模型结构摘要")
        table.add_column("层名称", style="cyan")
        table.add_column("类型", style="green")
        table.add_column("参数量", style="yellow")
        table.add_column("可训练", style="magenta")

        total_params = 0
        trainable_params = 0

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                num_params = sum(p.numel() for p in module.parameters())
                num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

                if num_params > 0:
                    table.add_row(
                        name or "root",
                        module.__class__.__name__,
                        f"{num_params:,}",
                        "是" if num_trainable > 0 else "否"
                    )
                    total_params += num_params
                    trainable_params += num_trainable

        console.print(table)
        console.print(f"\n总参数量: {total_params:,}")
        console.print(f"可训练参数: {trainable_params:,}")
        console.print(f"不可训练参数: {total_params - trainable_params:,}")

    @staticmethod
    def check_data_distribution(dataloader, num_batches=5):
        """检查数据分布"""
        console.print("\n[cyan]检查数据分布...[/cyan]")

        all_inputs = []
        all_targets = []

        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break
            all_inputs.append(inputs)
            all_targets.append(targets)

        inputs = torch.cat(all_inputs)
        targets = torch.cat(all_targets)

        console.print(f"输入形状: {inputs.shape}")
        console.print(f"输入范围: [{inputs.min():.4f}, {inputs.max():.4f}]")
        console.print(f"输入均值: {inputs.mean():.4f}")
        console.print(f"输入标准差: {inputs.std():.4f}")

        console.print(f"\n目标形状: {targets.shape}")
        console.print(f"目标范围: [{targets.min():.4f}, {targets.max():.4f}]")
        console.print(f"目标均值: {targets.mean():.4f}")
        console.print(f"目标标准差: {targets.std():.4f}")

        # 检查异常值
        if torch.isnan(inputs).any():
            console.print("[red]警告: 输入数据包含 NaN[/red]")
        if torch.isinf(inputs).any():
            console.print("[red]警告: 输入数据包含 Inf[/red]")

    @staticmethod
    def profile_training_step(model, dataloader, device='cpu'):
        """分析训练步骤性能"""
        console.print("\n[cyan]分析训练步骤性能...[/cyan]")

        import time

        model.to(device)
        model.train()

        # 获取一个batch
        inputs, targets = next(iter(dataloader))
        inputs, targets = inputs.to(device), targets.to(device)

        # 测量前向传播时间
        start = time.time()
        output = model(inputs)
        forward_time = time.time() - start

        # 测量损失计算时间
        start = time.time()
        loss = F.mse_loss(output, targets)
        loss_time = time.time() - start

        # 测量反向传播时间
        start = time.time()
        loss.backward()
        backward_time = time.time() - start

        total_time = forward_time + loss_time + backward_time

        table = Table(title="训练步骤性能分析")
        table.add_column("阶段", style="cyan")
        table.add_column("耗时 (ms)", style="green")
        table.add_column("占比", style="yellow")

        table.add_row("前向传播", f"{forward_time*1000:.2f}", f"{forward_time/total_time*100:.1f}%")
        table.add_row("损失计算", f"{loss_time*1000:.2f}", f"{loss_time/total_time*100:.1f}%")
        table.add_row("反向传播", f"{backward_time*1000:.2f}", f"{backward_time/total_time*100:.1f}%")
        table.add_row("总计", f"{total_time*1000:.2f}", "100%")

        console.print(table)


# ==================== 演示函数 ====================

def demo_training_monitoring():
    """演示训练监控"""
    console.print(Panel.fit("演示 1: 训练监控", style="bold magenta"))

    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    monitor = TrainingMonitor()

    # 模拟训练
    console.print("\n[cyan]开始训练...[/cyan]")
    for epoch in range(10):
        # 模拟训练指标
        train_loss = 1.0 / (epoch + 1) + np.random.rand() * 0.1
        val_loss = 1.2 / (epoch + 1) + np.random.rand() * 0.1
        accuracy = min(0.95, 0.5 + epoch * 0.05)

        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy
        }

        monitor.record(metrics)

        if epoch % 3 == 0:
            console.print(f"\nEpoch {epoch + 1}:")
            console.print(f"  训练损失: {train_loss:.4f}")
            console.print(f"  验证损失: {val_loss:.4f}")
            console.print(f"  准确率: {accuracy:.4f}")

            # 检查梯度健康
            x = torch.randn(4, 10)
            y = torch.randn(4, 1)
            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output, y)
            loss.backward()

            monitor.check_gradient_health(model)

    # 显示摘要
    console.print("\n")
    monitor.summary()
    monitor.save()


def demo_tensorboard_logging():
    """演示 TensorBoard 日志记录"""
    console.print(Panel.fit("演示 2: TensorBoard 日志记录", style="bold magenta"))

    logger = TensorBoardLogger()

    if logger.enabled:
        # 记录标量
        for step in range(100):
            logger.log_scalar('loss', 1.0 / (step + 1), step)
            logger.log_scalar('accuracy', min(0.99, step * 0.01), step)

        console.print("[green]已记录 100 步训练数据[/green]")
        console.print("运行以下命令查看: tensorboard --logdir=runs")

        logger.close()
    else:
        console.print("[yellow]TensorBoard 未启用，跳过演示[/yellow]")


def demo_diagnose_problems():
    """演示训练问题诊断"""
    console.print(Panel.fit("演示 3: 训练问题诊断", style="bold magenta"))

    # 创建有问题的模型（学习率过大）
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=10.0)  # 学习率过大
    early_stopping = EarlyStopping(patience=3)

    console.print("\n[cyan]训练有问题的模型...[/cyan]")

    for epoch in range(10):
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)

        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()

        console.print(f"\nEpoch {epoch + 1}: Loss = {loss.item():.4f}")

        # 检查梯度
        monitor = TrainingMonitor()
        grad_health = monitor.check_gradient_health(model)

        # 检查早停
        if early_stopping(loss.item()):
            break

        if torch.isnan(loss):
            console.print("[red]检测到 NaN 损失，训练终止[/red]")
            break


def demo_model_validation():
    """演示模型验证"""
    console.print(Panel.fit("演示 4: 模型验证", style="bold magenta"))

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    validator = ModelValidator()

    # 验证输出形状
    console.print("\n[cyan]1. 验证输出形状[/cyan]")
    validator.validate_output_shape(model, (4, 10), (4, 1))

    # 检查参数更新
    console.print("\n[cyan]2. 检查参数更新[/cyan]")
    validator.check_parameter_updates(model, optimizer, loss_fn)

    # 测试过拟合能力
    console.print("\n[cyan]3. 测试过拟合能力[/cyan]")
    validator.test_overfitting_capability(model, optimizer, loss_fn)


def demo_checkpoint_management():
    """演示检查点管理"""
    console.print(Panel.fit("演示 5: 检查点管理", style="bold magenta"))

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    manager = CheckpointManager(max_keep=3)

    # 保存几个检查点
    console.print("\n[cyan]保存检查点...[/cyan]")
    for epoch in range(5):
        metrics = {
            'loss': 1.0 / (epoch + 1),
            'accuracy': 0.5 + epoch * 0.1
        }
        manager.save(model, optimizer, epoch, metrics)

    # 加载最新检查点
    console.print("\n[cyan]加载最新检查点...[/cyan]")
    checkpoint = manager.load(model, optimizer)


def demo_debugging_tools():
    """演示调试工具"""
    console.print(Panel.fit("演示 6: 调试工具", style="bold magenta"))

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

    # 打印模型摘要
    console.print("\n[cyan]1. 模型结构摘要[/cyan]")
    DebugTools.print_model_summary(model)

    # 检查数据分布
    console.print("\n[cyan]2. 数据分布检查[/cyan]")
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32)
    DebugTools.check_data_distribution(dataloader)

    # 性能分析
    console.print("\n[cyan]3. 训练步骤性能分析[/cyan]")
    DebugTools.profile_training_step(model, dataloader)


def main():
    """主函数"""
    console.print(Panel.fit(
        "[bold cyan]深度学习框架 - 训练验证与调试工具[/bold cyan]\n"
        "验证驱动开发，闭环调试",
        style="bold blue"
    ))

    demos = [
        ("训练监控", demo_training_monitoring),
        ("TensorBoard 日志记录", demo_tensorboard_logging),
        ("训练问题诊断", demo_diagnose_problems),
        ("模型验证", demo_model_validation),
        ("检查点管理", demo_checkpoint_management),
        ("调试工具集", demo_debugging_tools),
    ]

    while True:
        console.print("\n[bold cyan]请选择演示:[/bold cyan]")
        for i, (name, _) in enumerate(demos, 1):
            console.print(f"  {i}. {name}")
        console.print("  0. 退出")

        try:
            choice = input("\n请输入选项 (0-6): ").strip()

            if choice == '0':
                console.print("[yellow]再见![/yellow]")
                break

            idx = int(choice) - 1
            if 0 <= idx < len(demos):
                console.print()
                demos[idx][1]()
            else:
                console.print("[red]无效选项，请重试[/red]")

        except KeyboardInterrupt:
            console.print("\n[yellow]已取消[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")


if __name__ == "__main__":
    main()
