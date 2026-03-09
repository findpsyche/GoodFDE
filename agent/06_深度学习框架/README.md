# 第六阶段：深度学习框架

> **核心目标**：掌握 PyTorch 基础、Transformers 架构、Hugging Face 生态和训练技巧，建立完整的模型训练和验证流程

## 学习目标

根据 [AI_Agent_Learning_Roadmap.md](../AI_Agent_Learning_Roadmap.md) 第六阶段规划：

- 掌握 PyTorch 的 Tensor 操作、自动微分、神经网络模块和训练循环
- 深入理解 Transformers 架构（Attention、Multi-head Attention、Position Encoding）
- 熟练使用 Hugging Face 生态（Transformers、Datasets、Tokenizers、Accelerate、PEFT）
- 掌握训练技巧（学习率调度、梯度累积、混合精度训练、分布式训练基础）
- 建立完整的训练验证和调试流程

## 📚 学习内容

### 核心技能

1. **PyTorch 基础**
   - Tensor 操作和数据类型
   - 自动微分（Autograd）
   - 神经网络模块（nn.Module）
   - 训练循环和优化器

2. **Transformers 架构**
   - Self-Attention 机制
   - Multi-head Attention
   - Position Encoding（绝对位置编码和相对位置编码）
   - Transformer Block（Encoder 和 Decoder）

3. **Hugging Face 生态**
   - Transformers 库（模型加载、推理、训练）
   - Datasets 库（数据加载和预处理）
   - Tokenizers（分词器使用和自定义）
   - Accelerate（分布式训练）和 PEFT（参数高效微调）

4. **训练技巧**
   - 学习率调度（Warmup、Cosine Annealing）
   - 梯度累积（模拟大 Batch Size）
   - 混合精度训练（FP16/BF16）
   - 分布式训练基础（DDP、FSDP）

5. **训练验证和调试**
   - 监控训练过程（Loss 曲线、学习率、梯度范数）
   - 诊断训练问题（Loss 不下降、过拟合、欠拟合）
   - 验证模型效果（验证集评估、错误分析）

## 设计哲学

本模块严格遵循 [AI_Agent_Learning_Roadmap.md](../AI_Agent_Learning_Roadmap.md) 的核心设计哲学：

### 1. 验证驱动

每个功能都强调"如何验证是否正确"：
- ✅ Tensor 操作：验证形状和数值
- ✅ Attention 机制：可视化注意力权重
- ✅ 模型训练：监控 Loss 曲线和验证集指标
- ✅ 生成质量：对比不同解码策略

### 2. 闭环思维

不是"写完就完"，而是"测试-修复-验证"循环：
- 实现 → 单元测试 → 发现问题 → 修复 → 再测试
- 训练 → 验证集评估 → 错误分析 → 调整 → 再训练
- 从"凭直觉调参"进化到"用实验定位问题"

### 3. 实战导向

关注真实场景中的坑和解决方案：
- Loss 不下降怎么办？
- 如何诊断过拟合？
- 如何优化训练速度？
- 如何验证模型是否学到了正确的模式？

## 📁 文件结构

```
06_深度学习框架/
├── 01_pytorch_basics.py          # PyTorch 基础
│   ├── Tensor 操作和数据类型
│   ├── 自动微分（Autograd）
│   ├── 神经网络模块（nn.Module）
│   └── 训练循环和优化器
│
├── 02_transformer_architecture.py # Transformers 架构
│   ├── Self-Attention 机制
│   ├── Multi-head Attention
│   ├── Position Encoding
│   └── Transformer Block
│
├── 03_huggingface_ecosystem.py   # Hugging Face 生态
│   ├── Transformers 库使用
│   ├── Datasets 库使用
│   ├── Tokenizers 使用
│   └── Accelerate 和 PEFT
│
├── 04_training_techniques.py     # 训练技巧
│   ├── 学习率调度
│   ├── 梯度累积
│   ├── 混合精度训练
│   └── 分布式训练基础
│
├── 05_training_validation.py     # 训练验证和调试
│   ├── 监控训练过程
│   ├── 诊断训练问题
│   └── 验证模型效果
│
├── examples/                     # 完整实践项目
│   ├── simple_transformer.py     # 从头实现 Transformer
│   ├── text_classification.py    # HF 文本分类
│   └── seq2seq.py               # Seq2Seq 任务
│
├── tests/                        # 测试文件
│   ├── test_pytorch_basics.py
│   ├── test_transformer.py
│   └── test_training.py
│
├── models/                       # 模型文件（不提交 Git）
│   └── README.md
│
├── README.md                     # 本文件
├── 快速开始指南.md
├── 学习笔记.md
├── requirements.txt
├── .env.example
└── .gitignore
```

## 🚀 快速开始

### 环境准备

**Python 版本要求**：Python 3.8+（推荐 3.10+）

**硬件要求**：
- CPU：任意（学习基础概念）
- GPU：推荐 NVIDIA GPU（CUDA 11.8+）用于训练加速
- 内存：至少 8GB RAM
- 磁盘：至少 10GB 空间（用于模型和数据集）

### 安装依赖

```bash
# 1. 创建虚拟环境（推荐）
conda create -n dl-framework python=3.10
conda activate dl-framework

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 2. 安装 PyTorch（根据你的 CUDA 版本选择）
# CPU 版本
pip install torch torchvision torchaudio

# GPU 版本（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU 版本（CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，配置必要的参数
# HUGGINGFACE_TOKEN=your_token_here  # 可选，用于下载私有模型
# WANDB_API_KEY=your_key_here        # 可选，用于实验跟踪
```

### 验证环境

```bash
# 运行测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_pytorch_basics.py -v

# 运行快速验证脚本
python 01_pytorch_basics.py
```

## 📖 学习路径

### 两周学习计划

#### Week 1: PyTorch 基础和 Transformers 架构

**Day 1-2: PyTorch 基础**
- [ ] 学习 Tensor 操作（创建、索引、切片、运算）
- [ ] 理解自动微分（Autograd）
- [ ] 实践：实现简单的线性回归
- [ ] 验证：测试梯度计算是否正确

**Day 3-4: 神经网络模块**
- [ ] 学习 nn.Module 和常用层（Linear、Conv2d、LSTM）
- [ ] 理解训练循环（前向传播、损失计算、反向传播、参数更新）
- [ ] 实践：实现 MLP 分类器
- [ ] 验证：在 MNIST 上训练并评估

**Day 5-6: Attention 机制**
- [ ] 理解 Self-Attention 原理（Q、K、V）
- [ ] 实现 Scaled Dot-Product Attention
- [ ] 实现 Multi-head Attention
- [ ] 验证：可视化注意力权重

**Day 7: Transformer Block**
- [ ] 理解 Position Encoding
- [ ] 实现完整的 Transformer Block
- [ ] 实践：从头实现简单 Transformer
- [ ] 验证：测试输入输出形状和数值

#### Week 2: Hugging Face 生态和训练技巧

**Day 8-9: Hugging Face 基础**
- [ ] 学习 Transformers 库（模型加载、推理）
- [ ] 学习 Datasets 库（数据加载、预处理）
- [ ] 学习 Tokenizers（分词器使用）
- [ ] 实践：使用预训练模型进行推理

**Day 10-11: 模型训练**
- [ ] 学习 Trainer API
- [ ] 学习训练技巧（学习率调度、梯度累积、混合精度）
- [ ] 实践：微调 BERT 进行文本分类
- [ ] 验证：监控训练过程，评估验证集

**Day 12-13: 高级训练技巧**
- [ ] 学习 PEFT（LoRA、Prefix Tuning）
- [ ] 学习 Accelerate（分布式训练）
- [ ] 实践：使用 LoRA 微调大模型
- [ ] 验证：对比微调前后的效果

**Day 14: Seq2Seq 任务**
- [ ] 理解 Encoder-Decoder 架构
- [ ] 学习解码策略（贪心、Beam Search）
- [ ] 实践：实现简单的翻译或摘要任务
- [ ] 验证：可视化 Attention 权重，评估生成质量

## 🎯 实践项目

### 项目 1：从头实现简单 Transformer

**目标**：深入理解 Transformer 架构的每个组件

**步骤**：
1. 实现 Scaled Dot-Product Attention
2. 实现 Multi-head Attention
3. 实现 Position Encoding（正弦余弦编码）
4. 实现 Feed-Forward Network
5. 实现 Transformer Block（带残差连接和 Layer Norm）
6. 组装完整的 Transformer 模型

**验证**：
- ✅ 单元测试每个组件（输入输出形状、数值正确性）
- ✅ 在小数据集上训练（如简单的序列复制任务）
- ✅ 可视化注意力权重
- ✅ 对比自己实现和 PyTorch 官方实现的输出

**成功标准**：
- 所有单元测试通过
- 模型能在简单任务上收敛
- 注意力权重符合预期（如关注相关位置）

### 项目 2：Hugging Face 文本分类

**目标**：熟练使用 Hugging Face 生态进行实际任务

**步骤**：
1. 选择数据集（如 IMDB 情感分类、AG News 新闻分类）
2. 加载预训练模型（如 BERT、RoBERTa）
3. 准备数据（Tokenization、数据增强）
4. 配置训练参数（学习率、Batch Size、Epochs）
5. 训练模型（使用 Trainer API）
6. 评估和分析（准确率、混淆矩阵、错误样本）

**验证**：
- ✅ 监控训练过程（Loss 曲线、验证集准确率）
- ✅ 使用 TensorBoard 或 wandb 可视化
- ✅ 在测试集上评估
- ✅ 分析错误样本（哪些样本预测错误，为什么）

**成功标准**：
- 验证集准确率达到合理水平（如 IMDB > 90%）
- 训练过程稳定（无 Loss 爆炸或不收敛）
- 能解释模型的预测（如使用 Attention 可视化）

### 项目 3：Seq2Seq 任务

**目标**：掌握序列到序列任务和生成式模型

**步骤**：
1. 选择任务（如机器翻译、文本摘要）
2. 准备数据（源语言和目标语言的对齐数据）
3. 选择模型（如 T5、BART、mT5）
4. 训练模型（使用 Seq2SeqTrainer）
5. 实现不同解码策略（贪心、Beam Search、Top-k Sampling）
6. 评估生成质量（BLEU、ROUGE）

**验证**：
- ✅ 对比不同解码策略的效果
- ✅ 可视化 Attention 权重（源序列和目标序列的对齐）
- ✅ 人工评估生成质量（流畅性、准确性）
- ✅ 测试边界情况（空输入、超长输入）

**成功标准**：
- 生成的文本流畅且符合任务要求
- BLEU/ROUGE 分数达到合理水平
- Attention 权重符合预期（如翻译时关注对应词）

## 📊 常用模型参考表

### Hugging Face 预训练模型

| 模型 | 参数量 | 适用任务 | 特点 |
|------|--------|----------|------|
| **BERT** | 110M (base), 340M (large) | 文本分类、NER、问答 | 双向编码器，适合理解任务 |
| **RoBERTa** | 125M (base), 355M (large) | 同 BERT | BERT 的改进版，性能更好 |
| **DistilBERT** | 66M | 同 BERT | BERT 的蒸馏版，速度快 2 倍 |
| **ALBERT** | 12M (base), 18M (large) | 同 BERT | 参数共享，模型更小 |
| **T5** | 60M (small) - 11B (xxl) | 文本生成、翻译、摘要 | 统一的 Text-to-Text 框架 |
| **BART** | 140M (base), 400M (large) | 文本生成、摘要 | Encoder-Decoder 架构 |
| **GPT-2** | 117M (small) - 1.5B (xl) | 文本生成 | 单向解码器，生成能力强 |
| **XLM-RoBERTa** | 270M (base), 550M (large) | 多语言任务 | 支持 100 种语言 |

### 中文模型

| 模型 | 参数量 | 适用任务 | 特点 |
|------|--------|----------|------|
| **BERT-Chinese** | 110M | 中文理解任务 | Google 官方中文 BERT |
| **RoBERTa-Chinese** | 102M | 中文理解任务 | 哈工大讯飞联合实验室 |
| **ERNIE** | 110M | 中文理解任务 | 百度，融入知识图谱 |
| **MacBERT** | 102M | 中文理解任务 | 哈工大，改进的中文 BERT |
| **Chinese-T5** | 220M | 中文生成任务 | T5 的中文版本 |

### 模型选择建议

**文本分类/NER**：
- 速度优先：DistilBERT
- 性能优先：RoBERTa-large
- 平衡：BERT-base

**文本生成**：
- 短文本：GPT-2
- 长文本：T5 或 BART
- 多语言：mT5

**中文任务**：
- 通用：RoBERTa-Chinese
- 知识密集：ERNIE
- 生成：Chinese-T5

## 🔧 常见问题

### PyTorch 相关

**Q: CUDA out of memory 错误**
```python
# 解决方案 1：减小 Batch Size
batch_size = 8  # 改为 4 或 2

# 解决方案 2：使用梯度累积
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 解决方案 3：清理缓存
torch.cuda.empty_cache()

# 解决方案 4：使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Q: 梯度消失或爆炸**
```python
# 解决方案 1：梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 解决方案 2：使用 Layer Normalization
nn.LayerNorm(hidden_size)

# 解决方案 3：调整学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 降低学习率
```

**Q: 训练速度慢**
```python
# 解决方案 1：使用 DataLoader 的多进程
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)

# 解决方案 2：使用混合精度训练（速度提升 2-3 倍）
from torch.cuda.amp import autocast
with autocast():
    output = model(input)

# 解决方案 3：使用编译模式（PyTorch 2.0+）
model = torch.compile(model)
```

### Hugging Face 相关

**Q: 模型下载慢或失败**
```bash
# 解决方案 1：使用镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 解决方案 2：手动下载模型
# 访问 https://huggingface.co/模型名称
# 下载所有文件到本地目录，然后加载
model = AutoModel.from_pretrained("./local_model_dir")

# 解决方案 3：使用代理
export HTTP_PROXY=http://your_proxy:port
export HTTPS_PROXY=http://your_proxy:port
```

**Q: Tokenizer 截断问题**
```python
# 问题：文本被截断，丢失信息
tokenizer(text, max_length=512, truncation=True)

# 解决方案 1：增加最大长度
tokenizer(text, max_length=1024, truncation=True)

# 解决方案 2：使用滑动窗口
def sliding_window_tokenize(text, tokenizer, max_length=512, stride=256):
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"][0]
    chunks = []
    for i in range(0, len(input_ids), stride):
        chunk = input_ids[i:i+max_length]
        chunks.append(chunk)
    return chunks

# 解决方案 3：使用长文本模型（如 Longformer、BigBird）
```

**Q: 训练 Loss 不下降**
```python
# 诊断步骤：
# 1. 检查数据是否正确
print(f"Sample input: {batch['input_ids'][0]}")
print(f"Sample label: {batch['labels'][0]}")

# 2. 检查学习率是否合适
# 太小：Loss 下降很慢
# 太大：Loss 震荡或爆炸
# 建议：从 1e-5 开始尝试

# 3. 检查是否冻结了不该冻结的层
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

# 4. 使用学习率查找器
from torch_lr_finder import LRFinder
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
lr_finder.plot()
```

### 训练相关

**Q: 过拟合（训练 Loss 下降但验证 Loss 上升）**
```python
# 解决方案 1：数据增强
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 解决方案 2：增加 Dropout
model.config.hidden_dropout_prob = 0.2
model.config.attention_probs_dropout_prob = 0.2

# 解决方案 3：Early Stopping
from transformers import EarlyStoppingCallback
trainer = Trainer(
    model=model,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 解决方案 4：减少训练轮数或使用更多数据
```

**Q: 欠拟合（训练和验证 Loss 都很高）**
```python
# 解决方案 1：增加模型容量
# 使用更大的模型（base → large）

# 解决方案 2：增加训练轮数
training_args.num_train_epochs = 10

# 解决方案 3：调整学习率
training_args.learning_rate = 5e-5  # 增加学习率

# 解决方案 4：检查数据质量
# 确保标签正确，数据清洗
```

## 📚 学习资源

### 官方文档

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [Hugging Face Course](https://huggingface.co/course)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [Datasets 文档](https://huggingface.co/docs/datasets)

### 推荐教程

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化理解 Transformer
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [BERT 论文](https://arxiv.org/abs/1810.04805) - 理解预训练语言模型
- [动手学深度学习](https://zh.d2l.ai/) - 中文深度学习教程

### 视频课程

- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Hugging Face 官方视频教程](https://www.youtube.com/c/HuggingFace)

### 实践资源

- [Hugging Face Model Hub](https://huggingface.co/models) - 预训练模型
- [Hugging Face Datasets Hub](https://huggingface.co/datasets) - 数据集
- [Papers with Code](https://paperswithcode.com/) - 论文和代码
- [PyTorch Examples](https://github.com/pytorch/examples) - 官方示例

## 🎓 学习检查清单

### PyTorch 基础
- [ ] 能创建和操作 Tensor（索引、切片、运算）
- [ ] 理解自动微分（Autograd）的原理
- [ ] 能使用 nn.Module 定义神经网络
- [ ] 能实现完整的训练循环
- [ ] 理解优化器和学习率调度器

### Transformers 架构
- [ ] 理解 Self-Attention 的计算过程（Q、K、V）
- [ ] 能实现 Scaled Dot-Product Attention
- [ ] 理解 Multi-head Attention 的作用
- [ ] 理解 Position Encoding 的必要性
- [ ] 能从头实现简单的 Transformer

### Hugging Face 生态
- [ ] 能加载和使用预训练模型
- [ ] 能使用 Datasets 库加载和预处理数据
- [ ] 理解 Tokenizer 的工作原理
- [ ] 能使用 Trainer API 训练模型
- [ ] 能使用 PEFT 进行参数高效微调

### 训练技巧
- [ ] 理解学习率调度的作用
- [ ] 能使用梯度累积模拟大 Batch Size
- [ ] 能使用混合精度训练加速
- [ ] 理解分布式训练的基本原理
- [ ] 能监控和诊断训练问题

### 验证和调试
- [ ] 能使用 TensorBoard 或 wandb 可视化训练过程
- [ ] 能诊断常见训练问题（Loss 不下降、过拟合）
- [ ] 能在验证集上评估模型
- [ ] 能进行错误分析
- [ ] 能可视化 Attention 权重

## 🚀 下一步

完成本阶段学习后，你应该能够：

✅ 熟练使用 PyTorch 构建和训练神经网络
✅ 深入理解 Transformers 架构的每个组件
✅ 使用 Hugging Face 生态进行实际 NLP 任务
✅ 掌握常用的训练技巧和优化方法
✅ 建立完整的训练验证和调试流程

### 继续学习

- **进阶主题**：
  - 大模型训练（分布式训练、模型并行）
  - 高级微调技术（LoRA、Adapter、Prefix Tuning）
  - 模型压缩（蒸馏、剪枝、量化）
  - 多模态模型（CLIP、BLIP）

- **实战项目**：
  - 构建自己的预训练语言模型
  - 实现 Instruction Tuning
  - 构建 RAG 系统（结合第三阶段知识）
  - 部署模型到生产环境（结合第五阶段知识）

- **研究方向**：
  - 阅读最新论文（arXiv、Papers with Code）
  - 复现 SOTA 模型
  - 参与开源项目（Hugging Face、PyTorch）

### 综合应用

结合前面阶段的知识，你可以：

1. **构建智能 Agent**（第二阶段 + 第六阶段）
   - 使用微调的模型作为 Agent 的大脑
   - 提高 Agent 的任务理解和执行能力

2. **优化 RAG 系统**（第三阶段 + 第六阶段）
   - 微调 Embedding 模型提高检索质量
   - 微调生成模型提高答案质量

3. **部署自己的模型**（第五阶段 + 第六阶段）
   - 训练自己的模型
   - 量化和优化
   - 部署到生产环境

---

**记住核心原则**：验证驱动、闭环思维、实战导向 🚀

每个功能都要验证，每个问题都要定位，每个实验都要记录！
