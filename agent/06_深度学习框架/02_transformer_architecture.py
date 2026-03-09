"""
Transformer 架构实现
从零实现完整的 Transformer 模型，包括自注意力机制、多头注意力、位置编码等核心组件
"""

import sys
from pathlib import Path

# 依赖检查
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
except ImportError as e:
    print(f"缺少必要的依赖库: {e}")
    print("请运行: pip install torch numpy matplotlib rich")
    sys.exit(1)

console = Console()


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    缩放点积注意力机制

    Args:
        query: [batch_size, seq_len, d_k]
        key: [batch_size, seq_len, d_k]
        value: [batch_size, seq_len, d_v]
        mask: [batch_size, seq_len, seq_len] 可选的掩码

    Returns:
        output: [batch_size, seq_len, d_v]
        attention_weights: [batch_size, seq_len, seq_len]
    """
    d_k = query.size(-1)

    # 计算注意力分数: Q * K^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)

    # 应用掩码（如果提供）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Softmax 归一化
    attention_weights = F.softmax(scores, dim=-1)

    # 加权求和
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """将最后一个维度拆分为 (num_heads, d_k)"""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """合并多头"""
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: 可选的掩码
        """
        batch_size = query.size(0)

        # 线性变换
        Q = self.W_q(query)  # [batch_size, seq_len_q, d_model]
        K = self.W_k(key)    # [batch_size, seq_len_k, d_model]
        V = self.W_v(value)  # [batch_size, seq_len_v, d_model]

        # 拆分多头
        Q = self.split_heads(Q)  # [batch_size, num_heads, seq_len_q, d_k]
        K = self.split_heads(K)  # [batch_size, num_heads, seq_len_k, d_k]
        V = self.split_heads(V)  # [batch_size, num_heads, seq_len_v, d_k]

        # 应用缩放点积注意力
        if mask is not None:
            mask = mask.unsqueeze(1)  # 为多头广播

        attn_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头
        attn_output = self.combine_heads(attn_output)  # [batch_size, seq_len_q, d_model]

        # 最终线性变换
        output = self.W_o(attn_output)

        return output, attention_weights


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
        """
        super().__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class FeedForward(nn.Module):
    """前馈神经网络"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout 比率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderBlock(nn.Module):
    """Transformer 编码器块"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout 比率
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: 可选的掩码
        """
        # 自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class DecoderBlock(nn.Module):
    """Transformer 解码器块"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout 比率
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model] 解码器输入
            encoder_output: [batch_size, src_len, d_model] 编码器输出
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（因果掩码）
        """
        # 自注意力（带因果掩码）
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 交叉注意力（编码器-解码器注意力）
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


class Transformer(nn.Module):
    """完整的 Transformer 模型"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_len=5000, dropout=0.1):
        """
        Args:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            d_ff: 前馈网络隐藏层维度
            max_len: 最大序列长度
            dropout: Dropout 比率
        """
        super().__init__()

        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 编码器
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # 解码器
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码（下三角矩阵）"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return ~mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: [batch_size, src_len] 源序列
            tgt: [batch_size, tgt_len] 目标序列
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        """
        # 嵌入 + 位置编码
        src_emb = self.dropout(self.pos_encoding(self.src_embedding(src) * np.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pos_encoding(self.tgt_embedding(tgt) * np.sqrt(self.d_model)))

        # 编码器
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)

        # 解码器
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)

        # 输出层
        output = self.fc_out(decoder_output)

        return output


def demo_attention_mechanism():
    """演示基础注意力机制"""
    console.print(Panel.fit("🔍 演示：缩放点积注意力机制", style="bold magenta"))

    # 创建示例数据
    batch_size, seq_len, d_k = 2, 4, 8
    query = torch.randn(batch_size, seq_len, d_k)
    key = torch.randn(batch_size, seq_len, d_k)
    value = torch.randn(batch_size, seq_len, d_k)

    console.print(f"\n输入维度:")
    console.print(f"  Query: {list(query.shape)}")
    console.print(f"  Key: {list(key.shape)}")
    console.print(f"  Value: {list(value.shape)}")

    # 计算注意力
    output, attention_weights = scaled_dot_product_attention(query, key, value)

    console.print(f"\n输出维度:")
    console.print(f"  Output: {list(output.shape)}")
    console.print(f"  Attention Weights: {list(attention_weights.shape)}")

    # 显示第一个样本的注意力权重
    console.print("\n第一个样本的注意力权重矩阵:")
    table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
    table.add_column("位置", style="dim")
    for i in range(seq_len):
        table.add_column(f"K{i}", justify="right")

    weights = attention_weights[0].detach().numpy()
    for i in range(seq_len):
        row = [f"Q{i}"] + [f"{weights[i, j]:.3f}" for j in range(seq_len)]
        table.add_row(*row)

    console.print(table)

    # 验证权重和为1
    row_sums = attention_weights.sum(dim=-1)
    console.print(f"\n✓ 每行权重和: {row_sums[0].tolist()} (应该都接近1.0)")


def demo_multihead_attention():
    """演示多头注意力"""
    console.print(Panel.fit("🎯 演示：多头注意力机制", style="bold magenta"))

    # 参数设置
    batch_size, seq_len, d_model = 2, 6, 512
    num_heads = 8

    # 创建模型和数据
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    console.print(f"\n模型参数:")
    console.print(f"  d_model: {d_model}")
    console.print(f"  num_heads: {num_heads}")
    console.print(f"  d_k (每个头的维度): {d_model // num_heads}")

    console.print(f"\n输入维度: {list(x.shape)}")

    # 前向传播
    output, attention_weights = mha(x, x, x)

    console.print(f"输出维度: {list(output.shape)}")
    console.print(f"注意力权重维度: {list(attention_weights.shape)}")
    console.print(f"  [batch_size, num_heads, seq_len, seq_len]")

    # 统计参数量
    total_params = sum(p.numel() for p in mha.parameters())
    console.print(f"\n总参数量: {total_params:,}")


def demo_position_encoding():
    """演示位置编码"""
    console.print(Panel.fit("📍 演示：位置编码", style="bold magenta"))

    d_model = 128
    max_len = 100

    # 创建位置编码
    pos_encoding = PositionalEncoding(d_model, max_len)

    # 创建示例输入
    batch_size, seq_len = 1, 50
    x = torch.zeros(batch_size, seq_len, d_model)

    # 应用位置编码
    output = pos_encoding(x)

    console.print(f"\n位置编码参数:")
    console.print(f"  d_model: {d_model}")
    console.print(f"  max_len: {max_len}")
    console.print(f"  输入维度: {list(x.shape)}")
    console.print(f"  输出维度: {list(output.shape)}")

    # 可视化位置编码
    pe_matrix = pos_encoding.pe[0, :seq_len, :].detach().numpy()

    plt.figure(figsize=(12, 6))
    plt.imshow(pe_matrix.T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='编码值')
    plt.xlabel('位置')
    plt.ylabel('维度')
    plt.title('位置编码可视化')
    plt.tight_layout()

    output_path = Path("position_encoding.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    console.print(f"\n✓ 位置编码可视化已保存到: {output_path}")


def demo_transformer_block():
    """演示 Transformer 编码器和解码器块"""
    console.print(Panel.fit("🧱 演示：Transformer 编码器/解码器块", style="bold magenta"))

    # 参数设置
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads, d_ff = 8, 2048

    # 创建编码器块
    encoder_block = EncoderBlock(d_model, num_heads, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)

    console.print("\n[bold cyan]编码器块:[/bold cyan]")
    console.print(f"  输入维度: {list(x.shape)}")

    encoder_output = encoder_block(x)
    console.print(f"  输出维度: {list(encoder_output.shape)}")

    encoder_params = sum(p.numel() for p in encoder_block.parameters())
    console.print(f"  参数量: {encoder_params:,}")

    # 创建解码器块
    decoder_block = DecoderBlock(d_model, num_heads, d_ff)
    tgt = torch.randn(batch_size, seq_len, d_model)

    # 创建因果掩码
    tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    tgt_mask = ~tgt_mask

    console.print("\n[bold cyan]解码器块:[/bold cyan]")
    console.print(f"  输入维度: {list(tgt.shape)}")
    console.print(f"  编码器输出维度: {list(encoder_output.shape)}")
    console.print(f"  因果掩码维度: {list(tgt_mask.shape)}")

    decoder_output = decoder_block(tgt, encoder_output, tgt_mask=tgt_mask)
    console.print(f"  输出维度: {list(decoder_output.shape)}")

    decoder_params = sum(p.numel() for p in decoder_block.parameters())
    console.print(f"  参数量: {decoder_params:,}")


def demo_full_transformer():
    """演示完整的 Transformer 模型"""
    console.print(Panel.fit("🚀 演示：完整 Transformer 模型", style="bold magenta"))

    # 参数设置
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048

    # 创建模型
    model = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, num_heads,
        num_encoder_layers, num_decoder_layers, d_ff
    )

    console.print("\n[bold cyan]模型架构:[/bold cyan]")
    console.print(f"  源词汇表大小: {src_vocab_size:,}")
    console.print(f"  目标词汇表大小: {tgt_vocab_size:,}")
    console.print(f"  模型维度: {d_model}")
    console.print(f"  注意力头数: {num_heads}")
    console.print(f"  编码器层数: {num_encoder_layers}")
    console.print(f"  解码器层数: {num_decoder_layers}")
    console.print(f"  前馈网络维度: {d_ff}")

    # 创建示例输入
    batch_size = 2
    src_len, tgt_len = 20, 15
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

    # 生成因果掩码
    tgt_mask = model.generate_square_subsequent_mask(tgt_len)

    console.print(f"\n[bold cyan]输入数据:[/bold cyan]")
    console.print(f"  源序列: {list(src.shape)}")
    console.print(f"  目标序列: {list(tgt.shape)}")
    console.print(f"  目标掩码: {list(tgt_mask.shape)}")

    # 前向传播
    output = model(src, tgt, tgt_mask=tgt_mask)

    console.print(f"\n[bold cyan]输出:[/bold cyan]")
    console.print(f"  输出维度: {list(output.shape)}")
    console.print(f"  [batch_size, tgt_len, tgt_vocab_size]")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    console.print(f"\n[bold cyan]模型统计:[/bold cyan]")
    console.print(f"  总参数量: {total_params:,}")
    console.print(f"  可训练参数: {trainable_params:,}")
    console.print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")


def demo_attention_visualization():
    """可视化注意力权重"""
    console.print(Panel.fit("📊 演示：注意力权重可视化", style="bold magenta"))

    # 创建一个简单的例子
    batch_size, seq_len, d_model = 1, 8, 64
    num_heads = 4

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    # 获取注意力权重
    _, attention_weights = mha(x, x, x)

    # 可视化所有头的注意力权重
    weights = attention_weights[0].detach().numpy()  # [num_heads, seq_len, seq_len]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(num_heads):
        ax = axes[i]
        im = ax.imshow(weights[i], cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'注意力头 {i+1}')
        ax.set_xlabel('Key 位置')
        ax.set_ylabel('Query 位置')

        # 添加数值标注
        for y in range(seq_len):
            for x_pos in range(seq_len):
                text = ax.text(x_pos, y, f'{weights[i, y, x_pos]:.2f}',
                             ha="center", va="center", color="white", fontsize=8)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    output_path = Path("attention_weights.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    console.print(f"\n✓ 注意力权重可视化已保存到: {output_path}")

    # 显示统计信息
    console.print("\n[bold cyan]注意力权重统计:[/bold cyan]")
    for i in range(num_heads):
        head_weights = weights[i]
        console.print(f"  头 {i+1}:")
        console.print(f"    最大值: {head_weights.max():.4f}")
        console.print(f"    最小值: {head_weights.min():.4f}")
        console.print(f"    平均值: {head_weights.mean():.4f}")
        console.print(f"    标准差: {head_weights.std():.4f}")


def main():
    """主函数"""
    console.print(Panel.fit(
        "[bold cyan]Transformer 架构实现[/bold cyan]\n"
        "从零实现完整的 Transformer 模型",
        border_style="cyan"
    ))

    demos = {
        "1": ("缩放点积注意力机制", demo_attention_mechanism),
        "2": ("多头注意力机制", demo_multihead_attention),
        "3": ("位置编码", demo_position_encoding),
        "4": ("Transformer 编码器/解码器块", demo_transformer_block),
        "5": ("完整 Transformer 模型", demo_full_transformer),
        "6": ("注意力权重可视化", demo_attention_visualization),
        "7": ("运行所有演示", None),
    }

    while True:
        console.print("\n[bold yellow]请选择演示:[/bold yellow]")
        for key, (desc, _) in demos.items():
            console.print(f"  {key}. {desc}")
        console.print("  0. 退出")

        choice = console.input("\n[bold green]请输入选项 (0-7): [/bold green]").strip()

        if choice == "0":
            console.print("[yellow]再见！[/yellow]")
            break
        elif choice == "7":
            for key in ["1", "2", "3", "4", "5", "6"]:
                console.print("\n" + "="*80 + "\n")
                demos[key][1]()
        elif choice in demos and demos[choice][1] is not None:
            console.print("\n" + "="*80 + "\n")
            demos[choice][1]()
        else:
            console.print("[red]无效选项，请重新选择！[/red]")


if __name__ == "__main__":
    main()
