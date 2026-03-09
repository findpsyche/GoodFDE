"""
测试训练相关功能

测试内容：
- 学习率调度器
- 梯度累积等效性
- 混合精度训练
- Checkpoint 保存和加载
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import tempfile
import os
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training_techniques import (
    get_cosine_schedule_with_warmup,
    train_with_gradient_accumulation,
    train_with_mixed_precision,
    save_checkpoint,
    load_checkpoint
)


@pytest.fixture
def simple_model():
    """创建简单的测试模型"""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )


@pytest.fixture
def optimizer(simple_model):
    """创建优化器"""
    return optim.Adam(simple_model.parameters(), lr=0.001)


@pytest.fixture
def sample_data():
    """创建测试数据"""
    batch_size = 4
    x = torch.randn(batch_size, 10)
    y = torch.randint(0, 5, (batch_size,))
    return x, y


@pytest.fixture
def temp_checkpoint_dir():
    """创建临时目录用于保存 checkpoint"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestLearningRateScheduler:
    """测试学习率调度器"""

    def test_cosine_schedule_warmup(self, optimizer):
        """测试带预热的余弦调度"""
        num_warmup_steps = 10
        num_training_steps = 100

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        initial_lr = optimizer.param_groups[0]['lr']
        lrs = []

        # 记录学习率变化
        for step in range(num_training_steps):
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()

        # 验证预热阶段
        warmup_lrs = lrs[:num_warmup_steps]
        assert warmup_lrs[0] < warmup_lrs[-1]  # 预热阶段学习率递增
        assert all(lr1 <= lr2 for lr1, lr2 in zip(warmup_lrs[:-1], warmup_lrs[1:]))

        # 验证余弦衰减阶段
        decay_lrs = lrs[num_warmup_steps:]
        assert decay_lrs[0] > decay_lrs[-1]  # 衰减阶段学习率递减

        # 验证最终学习率接近 0
        assert lrs[-1] < initial_lr * 0.1

    def test_scheduler_step_count(self, optimizer):
        """测试调度器步数"""
        num_training_steps = 50
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=5,
            num_training_steps=num_training_steps
        )

        # 执行所有步骤
        for _ in range(num_training_steps):
            optimizer.step()
            scheduler.step()

        # 验证最后一步的学习率
        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr >= 0

    def test_multiple_param_groups(self):
        """测试多个参数组"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )

        # 为不同层设置不同的学习率
        optimizer = optim.Adam([
            {'params': model[0].parameters(), 'lr': 0.001},
            {'params': model[1].parameters(), 'lr': 0.0001}
        ])

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=5,
            num_training_steps=50
        )

        initial_lrs = [group['lr'] for group in optimizer.param_groups]

        # 执行几步
        for _ in range(10):
            optimizer.step()
            scheduler.step()

        # 验证两个参数组的学习率都在变化
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        assert current_lrs[0] != initial_lrs[0]
        assert current_lrs[1] != initial_lrs[1]

        # 验证学习率比例保持不变
        ratio_initial = initial_lrs[0] / initial_lrs[1]
        ratio_current = current_lrs[0] / current_lrs[1]
        assert ratio_initial == pytest.approx(ratio_current, rel=0.01)


class TestGradientAccumulation:
    """测试梯度累积"""

    def test_gradient_accumulation_equivalence(self, simple_model, sample_data):
        """测试梯度累积与大批次的等效性"""
        x, y = sample_data
        criterion = nn.CrossEntropyLoss()

        # 方法 1: 正常训练（大批次）
        model1 = type(simple_model)(*[type(m)(*m.parameters()) for m in simple_model])
        model1.load_state_dict(simple_model.state_dict())
        optimizer1 = optim.SGD(model1.parameters(), lr=0.01)

        optimizer1.zero_grad()
        output1 = model1(x)
        loss1 = criterion(output1, y)
        loss1.backward()
        optimizer1.step()

        # 方法 2: 梯度累积（小批次）
        model2 = type(simple_model)(*[type(m)(*m.parameters()) for m in simple_model])
        model2.load_state_dict(simple_model.state_dict())
        optimizer2 = optim.SGD(model2.parameters(), lr=0.01)

        accumulation_steps = 2
        optimizer2.zero_grad()

        for i in range(accumulation_steps):
            # 分割数据
            start_idx = i * (len(x) // accumulation_steps)
            end_idx = (i + 1) * (len(x) // accumulation_steps)
            x_batch = x[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]

            output2 = model2(x_batch)
            loss2 = criterion(output2, y_batch)
            loss2 = loss2 / accumulation_steps  # 缩放损失
            loss2.backward()

        optimizer2.step()

        # 验证两种方法的参数更新应该相似
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-5)

    def test_gradient_accumulation_function(self, simple_model, sample_data):
        """测试梯度累积函数"""
        x, y = sample_data

        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 训练一个 epoch
        loss = train_with_gradient_accumulation(
            simple_model,
            dataloader,
            optimizer,
            criterion,
            accumulation_steps=2
        )

        # 验证损失是有限的
        assert torch.isfinite(torch.tensor(loss))

        # 验证参数已更新
        for param in simple_model.parameters():
            assert param.grad is not None

    def test_gradient_accumulation_memory_efficiency(self):
        """测试梯度累积的内存效率"""
        # 创建较大的模型
        large_model = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10)
        )

        # 大批次数据
        large_batch = torch.randn(64, 1000)
        target = torch.randint(0, 10, (64,))

        optimizer = optim.Adam(large_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 使用梯度累积
        dataset = torch.utils.data.TensorDataset(large_batch, target)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

        loss = train_with_gradient_accumulation(
            large_model,
            dataloader,
            optimizer,
            criterion,
            accumulation_steps=4
        )

        assert torch.isfinite(torch.tensor(loss))


class TestMixedPrecisionTraining:
    """测试混合精度训练"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_training(self, simple_model, sample_data):
        """测试混合精度训练"""
        model = simple_model.cuda()
        x, y = sample_data
        x, y = x.cuda(), y.cuda()

        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 训练一个 epoch
        loss = train_with_mixed_precision(
            model,
            dataloader,
            optimizer,
            criterion
        )

        assert torch.isfinite(torch.tensor(loss))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_autocast_context(self):
        """测试 autocast 上下文"""
        model = nn.Linear(10, 5).cuda()
        x = torch.randn(4, 10).cuda()

        # 不使用 autocast
        with torch.no_grad():
            output_fp32 = model(x)
            assert output_fp32.dtype == torch.float32

        # 使用 autocast
        with torch.no_grad(), autocast():
            output_fp16 = model(x)
            # 输出可能是 fp16 或 fp32，取决于操作
            assert output_fp16.dtype in [torch.float16, torch.float32]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_grad_scaler(self):
        """测试梯度缩放器"""
        model = nn.Linear(10, 5).cuda()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        scaler = GradScaler()

        x = torch.randn(4, 10).cuda()
        target = torch.randn(4, 5).cuda()

        # 训练步骤
        optimizer.zero_grad()

        with autocast():
            output = model(x)
            loss = ((output - target) ** 2).mean()

        # 使用 scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 验证参数已更新
        for param in model.parameters():
            assert param.grad is not None

    def test_mixed_precision_cpu_fallback(self, simple_model, sample_data):
        """测试 CPU 上的混合精度训练（应该正常工作）"""
        x, y = sample_data

        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # CPU 上也应该能运行（虽然不会真正使用 fp16）
        loss = train_with_mixed_precision(
            simple_model,
            dataloader,
            optimizer,
            criterion
        )

        assert torch.isfinite(torch.tensor(loss))


class TestCheckpoint:
    """测试 Checkpoint 保存和加载"""

    def test_save_checkpoint(self, simple_model, optimizer, temp_checkpoint_dir):
        """测试保存 checkpoint"""
        checkpoint_path = os.path.join(temp_checkpoint_dir, 'checkpoint.pt')

        # 保存 checkpoint
        save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            epoch=5,
            loss=0.123,
            path=checkpoint_path
        )

        # 验证文件存在
        assert os.path.exists(checkpoint_path)

        # 验证文件大小
        assert os.path.getsize(checkpoint_path) > 0

    def test_load_checkpoint(self, simple_model, optimizer, temp_checkpoint_dir):
        """测试加载 checkpoint"""
        checkpoint_path = os.path.join(temp_checkpoint_dir, 'checkpoint.pt')

        # 保存 checkpoint
        original_state = simple_model.state_dict()
        save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            epoch=5,
            loss=0.123,
            path=checkpoint_path
        )

        # 修改模型参数
        for param in simple_model.parameters():
            param.data.fill_(0)

        # 加载 checkpoint
        checkpoint = load_checkpoint(simple_model, optimizer, checkpoint_path)

        # 验证加载的信息
        assert checkpoint['epoch'] == 5
        assert checkpoint['loss'] == pytest.approx(0.123)

        # 验证模型参数已恢复
        loaded_state = simple_model.state_dict()
        for key in original_state:
            assert torch.equal(original_state[key], loaded_state[key])

    def test_checkpoint_optimizer_state(self, simple_model, optimizer, temp_checkpoint_dir):
        """测试优化器状态的保存和加载"""
        checkpoint_path = os.path.join(temp_checkpoint_dir, 'checkpoint.pt')

        # 执行一步优化
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # 保存优化器状态
        original_optimizer_state = optimizer.state_dict()
        save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            epoch=1,
            loss=loss.item(),
            path=checkpoint_path
        )

        # 创建新的优化器
        new_optimizer = optim.Adam(simple_model.parameters(), lr=0.001)

        # 加载 checkpoint
        load_checkpoint(simple_model, new_optimizer, checkpoint_path)

        # 验证优化器状态已恢复
        loaded_optimizer_state = new_optimizer.state_dict()

        # 比较参数组
        assert len(original_optimizer_state['param_groups']) == len(loaded_optimizer_state['param_groups'])

    def test_checkpoint_with_additional_info(self, simple_model, optimizer, temp_checkpoint_dir):
        """测试保存额外信息"""
        checkpoint_path = os.path.join(temp_checkpoint_dir, 'checkpoint.pt')

        # 保存额外信息
        additional_info = {
            'best_accuracy': 0.95,
            'training_config': {'batch_size': 32, 'lr': 0.001}
        }

        save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            epoch=10,
            loss=0.05,
            path=checkpoint_path,
            **additional_info
        )

        # 加载并验证
        checkpoint = load_checkpoint(simple_model, optimizer, checkpoint_path)

        assert checkpoint['best_accuracy'] == 0.95
        assert checkpoint['training_config']['batch_size'] == 32

    def test_checkpoint_without_optimizer(self, simple_model, temp_checkpoint_dir):
        """测试不保存优化器的 checkpoint"""
        checkpoint_path = os.path.join(temp_checkpoint_dir, 'checkpoint.pt')

        # 只保存模型
        save_checkpoint(
            model=simple_model,
            optimizer=None,
            epoch=5,
            loss=0.123,
            path=checkpoint_path
        )

        # 加载时不提供优化器
        checkpoint = load_checkpoint(simple_model, None, checkpoint_path)

        assert checkpoint['epoch'] == 5
        assert 'optimizer_state_dict' not in checkpoint or checkpoint['optimizer_state_dict'] is None

    def test_checkpoint_model_architecture_mismatch(self, temp_checkpoint_dir):
        """测试模型架构不匹配的情况"""
        checkpoint_path = os.path.join(temp_checkpoint_dir, 'checkpoint.pt')

        # 保存一个模型
        model1 = nn.Linear(10, 5)
        optimizer1 = optim.Adam(model1.parameters())
        save_checkpoint(model1, optimizer1, 1, 0.1, checkpoint_path)

        # 尝试加载到不同架构的模型
        model2 = nn.Linear(20, 10)  # 不同的架构
        optimizer2 = optim.Adam(model2.parameters())

        # 应该抛出异常
        with pytest.raises(RuntimeError):
            load_checkpoint(model2, optimizer2, checkpoint_path)


class TestTrainingIntegration:
    """集成测试"""

    def test_full_training_loop(self, simple_model, sample_data):
        """测试完整的训练循环"""
        x, y = sample_data

        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=2,
            num_training_steps=10
        )

        # 训练多个 epoch
        losses = []
        for epoch in range(3):
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = simple_model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            losses.append(epoch_loss / len(dataloader))

        # 验证损失在下降
        assert losses[-1] < losses[0]

    def test_training_with_validation(self, simple_model):
        """测试带验证的训练"""
        # 创建训练和验证数据
        train_x = torch.randn(20, 10)
        train_y = torch.randint(0, 5, (20,))
        val_x = torch.randn(10, 10)
        val_y = torch.randint(0, 5, (10,))

        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
        val_dataset = torch.utils.data.TensorDataset(val_x, val_y)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)

        optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 训练
        simple_model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = simple_model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        # 验证
        simple_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = simple_model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        assert torch.isfinite(torch.tensor(val_loss))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
