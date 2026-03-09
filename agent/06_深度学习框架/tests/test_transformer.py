"""
测试 Transformer 架构组件

测试内容：
- Attention 计算正确性
- Multi-Head Attention 输出形状
- Position Encoding 正确性
- Transformer Block 输出形状
"""

import pytest
import torch
import torch.nn as nn
import math
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transformer_architecture import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    PositionalEncoding,
    FeedForward,
    TransformerBlock,
    TransformerEncoder
)


@pytest.fixture
def batch_size():
    """批次大小"""
    return 2


@pytest.fixture
def seq_len():
    """序列长度"""
    return 10


@pytest.fixture
def d_model():
    """模型维度"""
    return 64


@pytest.fixture
def num_heads():
    """注意力头数"""
    return 8


@pytest.fixture
def sample_input(batch_size, seq_len, d_model):
    """创建测试输入"""
    return torch.randn(batch_size, seq_len, d_model)


@pytest.fixture
def attention_mask(batch_size, seq_len):
    """创建注意力掩码"""
    # 创建因果掩码（下三角）
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0).expand(batch_size, -1, -1)


class TestScaledDotProductAttention:
    """测试缩放点积注意力"""

    def test_attention_output_shape(self, batch_size, seq_len, d_model):
        """测试输出形状"""
        Q = torch.randn(batch_size, seq_len, d_model)
        K = torch.randn(batch_size, seq_len, d_model)
        V = torch.randn(batch_size, seq_len, d_model)

        output, attn_weights = scaled_dot_product_attention(Q, K, V)

        # 验证输出形状
        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, seq_len, seq_len)

    def test_attention_weights_sum(self, batch_size, seq_len, d_model):
        """测试注意力权重和为 1"""
        Q = torch.randn(batch_size, seq_len, d_model)
        K = torch.randn(batch_size, seq_len, d_model)
        V = torch.randn(batch_size, seq_len, d_model)

        _, attn_weights = scaled_dot_product_attention(Q, K, V)

        # 每行的权重和应该为 1
        weights_sum = attn_weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-6)

    def test_attention_with_mask(self, batch_size, seq_len, d_model, attention_mask):
        """测试带掩码的注意力"""
        Q = torch.randn(batch_size, seq_len, d_model)
        K = torch.randn(batch_size, seq_len, d_model)
        V = torch.randn(batch_size, seq_len, d_model)

        output, attn_weights = scaled_dot_product_attention(Q, K, V, mask=attention_mask)

        # 验证掩码位置的权重接近 0
        masked_positions = attention_mask[0]
        masked_weights = attn_weights[0][masked_positions]
        assert torch.all(masked_weights < 1e-6)

    def test_self_attention_identity(self):
        """测试自注意力的特殊情况"""
        # 当 Q=K=V 且只有一个 token 时，输出应该等于输入
        x = torch.randn(1, 1, 64)
        output, _ = scaled_dot_product_attention(x, x, x)

        assert torch.allclose(output, x, atol=1e-6)

    def test_attention_scaling(self, batch_size, seq_len):
        """测试缩放因子的影响"""
        d_k = 64
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # 验证缩放后的分数方差接近 1
        assert scores.var().item() == pytest.approx(1.0, rel=0.5)


class TestMultiHeadAttention:
    """测试多头注意力"""

    @pytest.fixture
    def mha(self, d_model, num_heads):
        """创建多头注意力模块"""
        return MultiHeadAttention(d_model, num_heads)

    def test_mha_output_shape(self, mha, sample_input):
        """测试输出形状"""
        output = mha(sample_input, sample_input, sample_input)

        # 输出形状应该与输入相同
        assert output.shape == sample_input.shape

    def test_mha_parameters(self, mha, d_model, num_heads):
        """测试参数数量和形状"""
        # W_q, W_k, W_v, W_o
        params = list(mha.parameters())
        assert len(params) == 8  # 4 个权重 + 4 个偏置

        # 验证投影矩阵形状
        assert mha.W_q.weight.shape == (d_model, d_model)
        assert mha.W_k.weight.shape == (d_model, d_model)
        assert mha.W_v.weight.shape == (d_model, d_model)
        assert mha.W_o.weight.shape == (d_model, d_model)

    def test_mha_with_mask(self, mha, sample_input, attention_mask):
        """测试带掩码的多头注意力"""
        output = mha(sample_input, sample_input, sample_input, mask=attention_mask)

        # 验证输出是有限的
        assert torch.all(torch.isfinite(output))

    def test_mha_cross_attention(self, mha, batch_size, seq_len, d_model):
        """测试交叉注意力"""
        query = torch.randn(batch_size, seq_len, d_model)
        key_value = torch.randn(batch_size, seq_len + 5, d_model)

        output = mha(query, key_value, key_value)

        # 输出形状应该与 query 相同
        assert output.shape == query.shape

    def test_mha_head_dimension(self, d_model, num_heads):
        """测试头维度必须整除"""
        # 正常情况
        mha = MultiHeadAttention(d_model, num_heads)
        assert mha.d_k == d_model // num_heads

        # 不能整除的情况应该报错
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=65, num_heads=8)


class TestPositionalEncoding:
    """测试位置编码"""

    @pytest.fixture
    def pos_encoding(self, d_model):
        """创建位置编码模块"""
        return PositionalEncoding(d_model, max_len=100)

    def test_pos_encoding_shape(self, pos_encoding, sample_input):
        """测试输出形状"""
        output = pos_encoding(sample_input)

        # 输出形状应该与输入相同
        assert output.shape == sample_input.shape

    def test_pos_encoding_values(self, d_model):
        """测试位置编码的值"""
        pos_encoding = PositionalEncoding(d_model, max_len=10)

        # 获取位置编码矩阵
        pe = pos_encoding.pe.squeeze(0)  # [max_len, d_model]

        # 验证第一个位置的编码
        first_pos = pe[0]
        assert first_pos[0].item() == pytest.approx(0.0)  # sin(0) = 0

        # 验证偶数维度使用 sin，奇数维度使用 cos
        for i in range(0, d_model, 2):
            # 偶数维度
            assert pe[1, i].item() != 0  # sin 不为 0

            # 奇数维度
            if i + 1 < d_model:
                assert pe[0, i + 1].item() == pytest.approx(1.0)  # cos(0) = 1

    def test_pos_encoding_different_lengths(self, pos_encoding, batch_size, d_model):
        """测试不同序列长度"""
        for seq_len in [5, 10, 20]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = pos_encoding(x)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_pos_encoding_max_len_exceeded(self, d_model):
        """测试超过最大长度"""
        max_len = 10
        pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        # 超过最大长度应该报错或截断
        x = torch.randn(1, max_len + 5, d_model)
        with pytest.raises((RuntimeError, IndexError)):
            pos_encoding(x)

    def test_pos_encoding_dropout(self, d_model):
        """测试 dropout"""
        pos_encoding = PositionalEncoding(d_model, dropout=0.5)
        pos_encoding.train()

        x = torch.randn(2, 10, d_model)
        output1 = pos_encoding(x)
        output2 = pos_encoding(x)

        # 训练模式下，dropout 应该导致输出不同
        assert not torch.equal(output1, output2)

        # 评估模式下，输出应该相同
        pos_encoding.eval()
        output3 = pos_encoding(x)
        output4 = pos_encoding(x)
        assert torch.equal(output3, output4)


class TestFeedForward:
    """测试前馈网络"""

    @pytest.fixture
    def ff(self, d_model):
        """创建前馈网络"""
        return FeedForward(d_model, d_ff=256, dropout=0.1)

    def test_ff_output_shape(self, ff, sample_input):
        """测试输出形状"""
        output = ff(sample_input)

        # 输出形状应该与输入相同
        assert output.shape == sample_input.shape

    def test_ff_parameters(self, ff, d_model):
        """测试参数"""
        # 两个线性层
        assert isinstance(ff.linear1, nn.Linear)
        assert isinstance(ff.linear2, nn.Linear)

        # 验证维度
        assert ff.linear1.in_features == d_model
        assert ff.linear1.out_features == 256
        assert ff.linear2.in_features == 256
        assert ff.linear2.out_features == d_model

    def test_ff_activation(self, ff, sample_input):
        """测试激活函数"""
        # 前向传播
        output = ff(sample_input)

        # 验证输出是有限的
        assert torch.all(torch.isfinite(output))

    def test_ff_dropout(self, d_model):
        """测试 dropout"""
        ff = FeedForward(d_model, d_ff=256, dropout=0.5)
        ff.train()

        x = torch.randn(2, 10, d_model)
        output1 = ff(x)
        output2 = ff(x)

        # 训练模式下应该不同
        assert not torch.equal(output1, output2)


class TestTransformerBlock:
    """测试 Transformer 块"""

    @pytest.fixture
    def transformer_block(self, d_model, num_heads):
        """创建 Transformer 块"""
        return TransformerBlock(d_model, num_heads, d_ff=256, dropout=0.1)

    def test_block_output_shape(self, transformer_block, sample_input):
        """测试输出形状"""
        output = transformer_block(sample_input)

        # 输出形状应该与输入相同
        assert output.shape == sample_input.shape

    def test_block_components(self, transformer_block):
        """测试组件"""
        assert isinstance(transformer_block.attention, MultiHeadAttention)
        assert isinstance(transformer_block.feed_forward, FeedForward)
        assert isinstance(transformer_block.norm1, nn.LayerNorm)
        assert isinstance(transformer_block.norm2, nn.LayerNorm)

    def test_block_with_mask(self, transformer_block, sample_input, attention_mask):
        """测试带掩码的 Transformer 块"""
        output = transformer_block(sample_input, mask=attention_mask)

        assert torch.all(torch.isfinite(output))

    def test_block_residual_connection(self, transformer_block, sample_input):
        """测试残差连接"""
        # 如果没有残差连接，输出会与输入差异很大
        output = transformer_block(sample_input)

        # 验证输出不是零
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_block_layer_norm(self, transformer_block, sample_input):
        """测试层归一化"""
        output = transformer_block(sample_input)

        # 验证输出的均值和方差
        # LayerNorm 应该使每个样本的特征维度均值接近 0，方差接近 1
        mean = output.mean(dim=-1)
        var = output.var(dim=-1)

        # 由于有残差连接，不会完全归一化，但应该在合理范围内
        assert torch.all(torch.abs(mean) < 2.0)
        assert torch.all(var > 0.1)


class TestTransformerEncoder:
    """测试 Transformer 编码器"""

    @pytest.fixture
    def encoder(self, d_model, num_heads):
        """创建 Transformer 编码器"""
        return TransformerEncoder(
            num_layers=3,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=256,
            dropout=0.1
        )

    def test_encoder_output_shape(self, encoder, sample_input):
        """测试输出形状"""
        output = encoder(sample_input)

        # 输出形状应该与输入相同
        assert output.shape == sample_input.shape

    def test_encoder_num_layers(self, encoder):
        """测试层数"""
        assert len(encoder.layers) == 3

        # 每一层都是 TransformerBlock
        for layer in encoder.layers:
            assert isinstance(layer, TransformerBlock)

    def test_encoder_with_mask(self, encoder, sample_input, attention_mask):
        """测试带掩码的编码器"""
        output = encoder(sample_input, mask=attention_mask)

        assert torch.all(torch.isfinite(output))

    def test_encoder_deep_network(self, d_model, num_heads):
        """测试深层网络"""
        # 创建更深的编码器
        deep_encoder = TransformerEncoder(
            num_layers=6,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=256,
            dropout=0.1
        )

        x = torch.randn(2, 10, d_model)
        output = deep_encoder(x)

        # 验证梯度可以流动
        loss = output.sum()
        loss.backward()

        # 检查第一层的梯度
        first_layer_params = list(deep_encoder.layers[0].parameters())
        assert all(p.grad is not None for p in first_layer_params)

    def test_encoder_gradient_flow(self, encoder, sample_input):
        """测试梯度流动"""
        output = encoder(sample_input)
        loss = output.sum()
        loss.backward()

        # 验证所有层都有梯度
        for layer in encoder.layers:
            for param in layer.parameters():
                assert param.grad is not None
                assert torch.all(torch.isfinite(param.grad))


class TestTransformerIntegration:
    """集成测试"""

    def test_full_forward_pass(self, batch_size, seq_len, d_model, num_heads):
        """测试完整的前向传播"""
        # 创建完整的 Transformer 编码器
        encoder = TransformerEncoder(
            num_layers=2,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=256,
            dropout=0.1
        )

        # 创建位置编码
        pos_encoding = PositionalEncoding(d_model)

        # 输入
        x = torch.randn(batch_size, seq_len, d_model)

        # 添加位置编码
        x = pos_encoding(x)

        # 编码
        output = encoder(x)

        # 验证输出
        assert output.shape == (batch_size, seq_len, d_model)
        assert torch.all(torch.isfinite(output))

    def test_training_step(self, batch_size, seq_len, d_model, num_heads):
        """测试训练步骤"""
        encoder = TransformerEncoder(
            num_layers=2,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=256,
            dropout=0.1
        )

        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

        # 训练一步
        x = torch.randn(batch_size, seq_len, d_model)
        target = torch.randn(batch_size, seq_len, d_model)

        optimizer.zero_grad()
        output = encoder(x)
        loss = ((output - target) ** 2).mean()
        loss.backward()
        optimizer.step()

        # 验证参数已更新
        assert all(p.grad is not None for p in encoder.parameters())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self, batch_size, seq_len, d_model, num_heads):
        """测试 CUDA 兼容性"""
        encoder = TransformerEncoder(
            num_layers=2,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=256,
            dropout=0.1
        ).cuda()

        x = torch.randn(batch_size, seq_len, d_model).cuda()
        output = encoder(x)

        assert output.device.type == 'cuda'
        assert torch.all(torch.isfinite(output))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
