"""
测试 PyTorch 基础功能

测试内容：
- Tensor 创建和操作
- 自动微分
- 简单模型的前向传播
- 设备管理
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pytorch_basics import (
    create_tensors,
    tensor_operations,
    autograd_example,
    SimpleNet,
    device_management
)


@pytest.fixture
def sample_tensor():
    """创建测试用的 Tensor"""
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture
def simple_model():
    """创建简单的测试模型"""
    return SimpleNet(input_size=10, hidden_size=20, output_size=5)


class TestTensorCreation:
    """测试 Tensor 创建"""

    def test_create_tensors(self):
        """测试各种 Tensor 创建方法"""
        tensors = create_tensors()

        assert 'zeros' in tensors
        assert 'ones' in tensors
        assert 'random' in tensors
        assert 'from_list' in tensors

        # 验证形状
        assert tensors['zeros'].shape == (3, 4)
        assert tensors['ones'].shape == (2, 3)
        assert tensors['random'].shape == (2, 2)

        # 验证值
        assert torch.all(tensors['zeros'] == 0)
        assert torch.all(tensors['ones'] == 1)
        assert torch.equal(tensors['from_list'], torch.tensor([[1, 2], [3, 4]]))

    def test_tensor_dtype(self):
        """测试 Tensor 数据类型"""
        float_tensor = torch.tensor([1.0, 2.0])
        int_tensor = torch.tensor([1, 2])

        assert float_tensor.dtype == torch.float32
        assert int_tensor.dtype == torch.int64

        # 类型转换
        converted = int_tensor.float()
        assert converted.dtype == torch.float32


class TestTensorOperations:
    """测试 Tensor 操作"""

    def test_basic_operations(self, sample_tensor):
        """测试基本运算"""
        results = tensor_operations(sample_tensor)

        # 加法
        expected_add = sample_tensor + 10
        assert torch.equal(results['add'], expected_add)

        # 乘法
        expected_mul = sample_tensor * 2
        assert torch.equal(results['mul'], expected_mul)

        # 矩阵乘法
        assert results['matmul'].shape == (2, 2)

    def test_tensor_indexing(self, sample_tensor):
        """测试索引和切片"""
        # 单个元素
        assert sample_tensor[0, 0].item() == 1.0

        # 切片
        first_row = sample_tensor[0, :]
        assert torch.equal(first_row, torch.tensor([1.0, 2.0]))

        # 布尔索引
        mask = sample_tensor > 2
        filtered = sample_tensor[mask]
        assert torch.equal(filtered, torch.tensor([3.0, 4.0]))

    def test_tensor_reshape(self):
        """测试形状变换"""
        x = torch.arange(12)

        # reshape
        reshaped = x.reshape(3, 4)
        assert reshaped.shape == (3, 4)

        # view
        viewed = x.view(2, 6)
        assert viewed.shape == (2, 6)

        # transpose
        transposed = reshaped.t()
        assert transposed.shape == (4, 3)

    def test_tensor_concatenation(self):
        """测试拼接操作"""
        x = torch.ones(2, 3)
        y = torch.zeros(2, 3)

        # 按行拼接
        cat_dim0 = torch.cat([x, y], dim=0)
        assert cat_dim0.shape == (4, 3)

        # 按列拼接
        cat_dim1 = torch.cat([x, y], dim=1)
        assert cat_dim1.shape == (2, 6)


class TestAutograd:
    """测试自动微分"""

    def test_basic_autograd(self):
        """测试基本的自动微分"""
        x = torch.tensor([2.0], requires_grad=True)
        y = x ** 2 + 3 * x + 1

        y.backward()

        # dy/dx = 2x + 3 = 2*2 + 3 = 7
        assert x.grad.item() == pytest.approx(7.0)

    def test_autograd_example(self):
        """测试自动微分示例"""
        results = autograd_example()

        assert 'x' in results
        assert 'y' in results
        assert 'grad' in results

        # 验证梯度计算
        x_val = results['x'].item()
        expected_grad = 2 * x_val
        assert results['grad'].item() == pytest.approx(expected_grad)

    def test_gradient_accumulation(self):
        """测试梯度累积"""
        x = torch.tensor([1.0], requires_grad=True)

        # 第一次反向传播
        y1 = x ** 2
        y1.backward()
        grad1 = x.grad.clone()

        # 第二次反向传播（不清零）
        y2 = x ** 3
        y2.backward()

        # 梯度应该累积
        assert x.grad.item() == grad1.item() + 3.0

    def test_no_grad_context(self):
        """测试 no_grad 上下文"""
        x = torch.tensor([1.0], requires_grad=True)

        with torch.no_grad():
            y = x ** 2

        # 在 no_grad 中计算的结果不应该有梯度
        assert not y.requires_grad


class TestSimpleModel:
    """测试简单神经网络模型"""

    def test_model_creation(self, simple_model):
        """测试模型创建"""
        assert isinstance(simple_model.fc1, nn.Linear)
        assert isinstance(simple_model.fc2, nn.Linear)
        assert isinstance(simple_model.relu, nn.ReLU)

        # 验证层的维度
        assert simple_model.fc1.in_features == 10
        assert simple_model.fc1.out_features == 20
        assert simple_model.fc2.in_features == 20
        assert simple_model.fc2.out_features == 5

    def test_forward_pass(self, simple_model):
        """测试前向传播"""
        batch_size = 4
        x = torch.randn(batch_size, 10)

        output = simple_model(x)

        # 验证输出形状
        assert output.shape == (batch_size, 5)

        # 验证输出是有限的数值
        assert torch.all(torch.isfinite(output))

    def test_model_parameters(self, simple_model):
        """测试模型参数"""
        params = list(simple_model.parameters())

        # 应该有 4 个参数：fc1.weight, fc1.bias, fc2.weight, fc2.bias
        assert len(params) == 4

        # 验证参数形状
        assert params[0].shape == (20, 10)  # fc1.weight
        assert params[1].shape == (20,)     # fc1.bias
        assert params[2].shape == (5, 20)   # fc2.weight
        assert params[3].shape == (5,)      # fc2.bias

    def test_model_training_mode(self, simple_model):
        """测试训练/评估模式切换"""
        # 默认是训练模式
        assert simple_model.training

        # 切换到评估模式
        simple_model.eval()
        assert not simple_model.training

        # 切换回训练模式
        simple_model.train()
        assert simple_model.training

    def test_gradient_flow(self, simple_model):
        """测试梯度流动"""
        x = torch.randn(2, 10)
        target = torch.randn(2, 5)

        # 前向传播
        output = simple_model(x)
        loss = ((output - target) ** 2).mean()

        # 反向传播
        loss.backward()

        # 验证所有参数都有梯度
        for param in simple_model.parameters():
            assert param.grad is not None
            assert torch.all(torch.isfinite(param.grad))


class TestDeviceManagement:
    """测试设备管理"""

    def test_cpu_device(self):
        """测试 CPU 设备"""
        x = torch.tensor([1.0, 2.0])
        assert x.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """测试 CUDA 设备"""
        x = torch.tensor([1.0, 2.0]).cuda()
        assert x.device.type == 'cuda'

        # 移回 CPU
        x_cpu = x.cpu()
        assert x_cpu.device.type == 'cpu'

    def test_device_management_function(self):
        """测试设备管理函数"""
        results = device_management()

        assert 'device' in results
        assert 'tensor_device' in results

        # 验证设备类型
        device = results['device']
        assert device.type in ['cpu', 'cuda']

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_to_device(self, simple_model):
        """测试模型移动到设备"""
        # 移动到 CUDA
        simple_model.cuda()

        # 验证所有参数都在 CUDA 上
        for param in simple_model.parameters():
            assert param.device.type == 'cuda'

        # 移回 CPU
        simple_model.cpu()

        for param in simple_model.parameters():
            assert param.device.type == 'cpu'


class TestTensorMemory:
    """测试 Tensor 内存管理"""

    def test_inplace_operations(self):
        """测试原地操作"""
        x = torch.tensor([1.0, 2.0, 3.0])
        x_id = id(x)

        # 原地操作
        x.add_(1.0)

        # 验证是同一个对象
        assert id(x) == x_id
        assert torch.equal(x, torch.tensor([2.0, 3.0, 4.0]))

    def test_tensor_clone(self):
        """测试 Tensor 克隆"""
        x = torch.tensor([1.0, 2.0])
        y = x.clone()

        # 修改 y 不应该影响 x
        y.add_(1.0)

        assert torch.equal(x, torch.tensor([1.0, 2.0]))
        assert torch.equal(y, torch.tensor([2.0, 3.0]))

    def test_tensor_detach(self):
        """测试 detach"""
        x = torch.tensor([1.0], requires_grad=True)
        y = x ** 2

        # detach 后不再追踪梯度
        z = y.detach()
        assert not z.requires_grad

        # 但共享数据
        assert z.data_ptr() == y.data_ptr()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
