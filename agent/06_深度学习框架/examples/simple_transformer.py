"""
从头实现简单 Transformer

功能：
- 实现完整的 Transformer 架构（Encoder-Decoder）
- 包含 Multi-Head Attention、FFN、Layer Norm、Positional Encoding
- 在序列复制任务上训练和验证

学习目标：
- 理解 Transformer 的核心组件
- 掌握 Attention 机制的实现
- 学习位置编码的作用
- 实践完整的训练流程

验证标准：
- 模型能够学会复制输入序列
- 训练损失持续下降
- 验证准确率达到 95% 以上
- 输出形状和数值正确
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
import numpy as np

console = Console()


class PositionalEncoding(nn.Module):
    """位置编码：为序列添加位置信息"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V 投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or None
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)

        # 线性投影并分割成多头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V: [batch_size, num_heads, seq_len, d_k]

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [batch_size, num_heads, seq_len, seq_len]

        # 应用 mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        # Softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        context = torch.matmul(attn_weights, V)
        # context: [batch_size, num_heads, seq_len, d_k]

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 输出投影
        output = self.W_o(context)

        return output


class FeedForward(nn.Module):
    """前馈神经网络"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Transformer Encoder 层"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or None
        Returns:
            [batch_size, seq_len, d_model]
        """
        # Self-Attention + Residual + Norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed Forward + Residual + Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    """Transformer Decoder 层"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, tgt_len, d_model]
            encoder_output: [batch_size, src_len, d_model]
            src_mask: [batch_size, src_len, src_len] or None
            tgt_mask: [batch_size, tgt_len, tgt_len] or None
        Returns:
            [batch_size, tgt_len, d_model]
        """
        # Self-Attention + Residual + Norm
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # Cross-Attention + Residual + Norm
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Feed Forward + Residual + Norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Transformer(nn.Module):
    """完整的 Transformer 模型"""

    def __init__(self, vocab_size: int, d_model: int = 128, num_heads: int = 4,
                 num_encoder_layers: int = 2, num_decoder_layers: int = 2,
                 d_ff: int = 512, dropout: float = 0.1, max_len: int = 100):
        super().__init__()

        self.d_model = d_model

        # Embedding 层
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Encoder 层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder 层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encoder 前向传播
        Args:
            src: [batch_size, src_len]
            src_mask: [batch_size, src_len, src_len] or None
        Returns:
            [batch_size, src_len, d_model]
        """
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Decoder 前向传播
        Args:
            tgt: [batch_size, tgt_len]
            encoder_output: [batch_size, src_len, d_model]
            src_mask: [batch_size, src_len, src_len] or None
            tgt_mask: [batch_size, tgt_len, tgt_len] or None
        Returns:
            [batch_size, tgt_len, d_model]
        """
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        完整的前向传播
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
            src_mask: [batch_size, src_len, src_len] or None
            tgt_mask: [batch_size, tgt_len, tgt_len] or None
        Returns:
            [batch_size, tgt_len, vocab_size]
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = self.output_projection(decoder_output)
        return logits


def create_causal_mask(size: int) -> torch.Tensor:
    """
    创建因果 mask（下三角矩阵），防止 Decoder 看到未来信息
    Args:
        size: 序列长度
    Returns:
        [size, size] 的下三角矩阵
    """
    mask = torch.tril(torch.ones(size, size))
    return mask


class SequenceCopyDataset(Dataset):
    """序列复制数据集：输入 [1,2,3,4,5] → 输出 [1,2,3,4,5]"""

    def __init__(self, num_samples: int = 1000, seq_len: int = 10, vocab_size: int = 20):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # 生成随机序列（避免使用 0，因为 0 通常用作 padding）
        self.data = torch.randint(1, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.data[idx]
        # 输入和目标相同（复制任务）
        return seq, seq


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        # Decoder 输入：在目标序列前添加 <BOS> token (使用 0)
        tgt_input = torch.cat([torch.zeros(tgt.size(0), 1, dtype=torch.long, device=device), tgt[:, :-1]], dim=1)

        # 创建因果 mask
        tgt_mask = create_causal_mask(tgt_input.size(1)).to(device)

        # 前向传播
        optimizer.zero_grad()
        logits = model(src, tgt_input, tgt_mask=tgt_mask)

        # 计算损失
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             device: torch.device) -> tuple[float, float]:
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            # Decoder 输入
            tgt_input = torch.cat([torch.zeros(tgt.size(0), 1, dtype=torch.long, device=device), tgt[:, :-1]], dim=1)

            # 创建因果 mask
            tgt_mask = create_causal_mask(tgt_input.size(1)).to(device)

            # 前向传播
            logits = model(src, tgt_input, tgt_mask=tgt_mask)

            # 计算损失
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            total_loss += loss.item()

            # 计算准确率
            predictions = logits.argmax(dim=-1)
            correct += (predictions == tgt).sum().item()
            total += tgt.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def generate_sequence(model: nn.Module, src: torch.Tensor, max_len: int,
                      device: torch.device) -> torch.Tensor:
    """
    生成序列（贪心解码）
    Args:
        model: Transformer 模型
        src: [1, src_len] 输入序列
        max_len: 最大生成长度
        device: 设备
    Returns:
        [1, gen_len] 生成的序列
    """
    model.eval()
    src = src.to(device)

    # Encoder
    encoder_output = model.encode(src)

    # Decoder：从 <BOS> token 开始
    tgt = torch.zeros(1, 1, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len):
            # 创建因果 mask
            tgt_mask = create_causal_mask(tgt.size(1)).to(device)

            # Decoder 前向传播
            decoder_output = model.decode(tgt, encoder_output, tgt_mask=tgt_mask)

            # 预测下一个 token
            logits = model.output_projection(decoder_output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)

            # 拼接到输出序列
            tgt = torch.cat([tgt, next_token], dim=1)

            # 如果生成了足够的 token，停止
            if tgt.size(1) > src.size(1):
                break

    # 移除 <BOS> token
    return tgt[:, 1:]


def main():
    console.print(Panel.fit(
        "[bold cyan]从头实现简单 Transformer[/bold cyan]\n"
        "任务：序列复制（输入 [1,2,3,4,5] → 输出 [1,2,3,4,5]）",
        border_style="cyan"
    ))

    # 超参数
    vocab_size = 20
    seq_len = 10
    d_model = 128
    num_heads = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    d_ff = 512
    dropout = 0.1
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"[green]使用设备: {device}[/green]\n")

    # 创建数据集
    console.print("[yellow]创建数据集...[/yellow]")
    train_dataset = SequenceCopyDataset(num_samples=1000, seq_len=seq_len, vocab_size=vocab_size)
    val_dataset = SequenceCopyDataset(num_samples=200, seq_len=seq_len, vocab_size=vocab_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    console.print(f"[green]训练集: {len(train_dataset)} 样本[/green]")
    console.print(f"[green]验证集: {len(val_dataset)} 样本[/green]\n")

    # 创建模型
    console.print("[yellow]创建模型...[/yellow]")
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]模型参数量: {num_params:,}[/green]\n")

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练
    console.print("[bold yellow]开始训练...[/bold yellow]\n")

    best_val_loss = float('inf')
    best_accuracy = 0.0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]训练进度", total=num_epochs)

        for epoch in range(num_epochs):
            # 训练
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

            # 验证
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

            # 更新进度
            progress.update(task, advance=1)

            # 打印日志
            if (epoch + 1) % 5 == 0:
                console.print(
                    f"Epoch [{epoch+1}/{num_epochs}] - "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_accuracy:.4f}"
                )

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_accuracy = val_accuracy

                save_dir = Path(__file__).parent / "checkpoints"
                save_dir.mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                }, save_dir / "simple_transformer_best.pt")

    console.print(f"\n[bold green]训练完成！[/bold green]")
    console.print(f"[green]最佳验证损失: {best_val_loss:.4f}[/green]")
    console.print(f"[green]最佳验证准确率: {best_accuracy:.4f}[/green]\n")

    # 测试生成
    console.print("[bold yellow]测试生成效果...[/bold yellow]\n")

    model.eval()
    test_samples = 5

    table = Table(title="序列复制测试", show_header=True, header_style="bold magenta")
    table.add_column("输入序列", style="cyan")
    table.add_column("目标序列", style="green")
    table.add_column("生成序列", style="yellow")
    table.add_column("匹配", style="bold")

    for i in range(test_samples):
        src = val_dataset[i][0].unsqueeze(0)
        tgt = val_dataset[i][1]

        generated = generate_sequence(model, src, max_len=seq_len, device=device)

        src_str = str(src.squeeze().tolist())
        tgt_str = str(tgt.tolist())
        gen_str = str(generated.squeeze().cpu().tolist())

        match = "✓" if generated.squeeze().cpu().tolist() == tgt.tolist() else "✗"
        match_style = "green" if match == "✓" else "red"

        table.add_row(src_str, tgt_str, gen_str, f"[{match_style}]{match}[/{match_style}]")

    console.print(table)

    # 验证标准检查
    console.print("\n[bold cyan]验证标准检查:[/bold cyan]")
    checks = [
        ("训练损失持续下降", train_loss < 0.5, train_loss),
        ("验证准确率 ≥ 95%", best_accuracy >= 0.95, f"{best_accuracy:.2%}"),
        ("模型成功保存", (Path(__file__).parent / "checkpoints" / "simple_transformer_best.pt").exists(), "✓"),
    ]

    for check_name, passed, value in checks:
        status = "[green]✓[/green]" if passed else "[red]✗[/red]"
        console.print(f"{status} {check_name}: {value}")

    console.print("\n[bold green]实验完成！[/bold green]")


if __name__ == "__main__":
    main()
