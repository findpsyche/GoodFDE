# 第五阶段：模型部署与调优

> **学习目标**：掌握本地模型部署、量化、微调和推理优化，建立完整的模型评估和验证流程

## 📚 学习内容

### 核心技能
- ✅ 使用 Ollama 和 vLLM 部署本地模型
- ✅ 理解和应用模型量化（GGUF、GPTQ、AWQ）
- ✅ 使用 LoRA/QLoRA 进行模型微调
- ✅ 掌握推理优化技术（批处理、KV Cache、Flash Attention）
- ✅ 建立完整的部署验证和性能测试流程
- ✅ 对比云端 API 与本地部署的成本效益

### 设计哲学
本阶段强调**验证驱动**和**闭环思维**：
- 🔍 每个部署都要验证是否正常工作
- 📊 每个优化都要测量性能提升
- 🔄 部署 → 测试 → 优化 → 再测试
- 💰 关注成本效益，不盲目追求性能

## 📁 文件结构

```
05_模型部署与调优/
├── README.md                          # 本文件
├── 快速开始指南.md                    # 快速上手指南
├── 学习笔记.md                        # 学习笔记模板
├── requirements.txt                   # Python 依赖
├── .env.example                       # 环境变量示例
├── .gitignore                         # Git 忽略文件
│
├── 01_ollama_deployment.py            # Ollama 部署示例
├── 02_vllm_deployment.py              # vLLM 部署示例
├── 03_quantization_comparison.py     # 量化方法对比
├── 04_lora_finetuning.py             # LoRA 微调示例
├── 05_inference_optimization.py      # 推理优化示例
├── 06_performance_benchmark.py       # 性能基准测试
├── 07_cost_analysis.py               # 成本效益分析
│
├── examples/                          # 完整示例
│   ├── chatbot_local.py              # 本地聊天机器人
│   ├── api_server.py                 # FastAPI 服务封装
│   └── production_deploy.py          # 生产环境部署示例
│
├── scripts/                           # 实用脚本
│   ├── download_models.sh            # 模型下载脚本
│   ├── test_deployment.py            # 部署测试脚本
│   ├── monitor_performance.py        # 性能监控脚本
│   └── compare_models.py             # 模型对比脚本
│
├── tests/                             # 测试文件
│   ├── test_ollama.py                # Ollama 测试
│   ├── test_vllm.py                  # vLLM 测试
│   └── test_finetuning.py            # 微调测试
│
└── models/                            # 模型文件（不提交到 Git）
    └── .gitkeep
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n model-deploy python=3.10
conda activate model-deploy

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入必要的配置
```

### 2. 安装 Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# 从 https://ollama.com/download 下载安装包

# 验证安装
ollama --version

# 拉取模型
ollama pull llama3:8b
```

### 3. 运行第一个示例

```bash
# 测试 Ollama 部署
python 01_ollama_deployment.py

# 测试 vLLM 部署（需要 GPU）
python 02_vllm_deployment.py

# 量化对比
python 03_quantization_comparison.py
```

## 📖 学习路径

### 第一周：本地部署基础

#### Day 1-2: Ollama 入门
- [ ] 阅读 `01_ollama_deployment.py`
- [ ] 部署 Llama 3 8B 模型
- [ ] 测试基本推理功能
- [ ] 记录性能指标

#### Day 3-4: vLLM 高性能部署
- [ ] 阅读 `02_vllm_deployment.py`
- [ ] 对比 Ollama vs vLLM 性能
- [ ] 测试并发处理能力
- [ ] 优化配置参数

#### Day 5-7: 量化技术
- [ ] 阅读 `03_quantization_comparison.py`
- [ ] 测试不同量化级别（Q4/Q5/Q6）
- [ ] 对比速度、内存、质量
- [ ] 选择最优配置

### 第二周：微调与优化

#### Day 8-10: LoRA 微调
- [ ] 阅读 `04_lora_finetuning.py`
- [ ] 准备训练数据
- [ ] 训练 LoRA 模型
- [ ] 评估微调效果

#### Day 11-12: 推理优化
- [ ] 阅读 `05_inference_optimization.py`
- [ ] 测试批处理优化
- [ ] 测试 Flash Attention
- [ ] 测量性能提升

#### Day 13-14: 综合实践
- [ ] 运行 `06_performance_benchmark.py`
- [ ] 运行 `07_cost_analysis.py`
- [ ] 完成成本效益分析报告
- [ ] 总结最佳实践

## 🎯 实践项目

### 项目 1：本地聊天机器人
**目标**：部署一个完整的本地聊天机器人

**步骤**：
1. 使用 Ollama 部署 Llama 3 8B
2. 实现对话历史管理
3. 添加流式响应
4. 测试性能和质量

**验证**：
- ✅ 响应速度 < 2 秒
- ✅ 支持多轮对话
- ✅ 内存占用 < 8GB

### 项目 2：模型微调
**目标**：微调一个特定任务的模型

**步骤**：
1. 选择任务（代码注释、SQL 生成等）
2. 准备 200-500 条训练数据
3. 使用 QLoRA 微调
4. 评估和部署

**验证**：
- ✅ 训练 loss 正常下降
- ✅ 验证集性能提升 > 20%
- ✅ 泛化能力良好

### 项目 3：生产环境部署
**目标**：部署一个生产级的推理服务

**步骤**：
1. 使用 FastAPI 封装服务
2. 添加限流和缓存
3. 实现监控和日志
4. Docker 容器化

**验证**：
- ✅ 支持并发请求
- ✅ 有健康检查接口
- ✅ 有完整的监控指标

## 📊 性能基准

### 硬件配置参考

| 模型 | 量化 | VRAM | 速度 (tokens/s) | 适用场景 |
|------|------|------|-----------------|----------|
| Llama 3 8B | Q4 | 4GB | 30-50 | 实时对话 |
| Llama 3 8B | Q5 | 5GB | 25-40 | 平衡质量 |
| Llama 3 70B | Q4 | 40GB | 5-10 | 复杂推理 |
| Mistral 7B | Q4 | 4GB | 40-60 | 快速响应 |

### 成本对比（示例）

| 方案 | 初始成本 | 月度成本 | 适用场景 |
|------|----------|----------|----------|
| OpenAI API | $0 | $50-500 | 低频使用 |
| 本地 RTX 3090 | $1500 | $20 (电费) | 中高频使用 |
| 云端 GPU (A100) | $0 | $300-1000 | 高频使用 |

## 🔧 常见问题

### Q1: 内存不足怎么办？
**A**:
- 使用更激进的量化（Q4 → Q3）
- 减少 `max_model_len`
- 使用更小的模型

### Q2: 推理速度慢怎么办？
**A**:
- 切换到 vLLM
- 启用 Flash Attention
- 使用 GPU 而非 CPU

### Q3: 输出质量差怎么办？
**A**:
- 使用更高精度的量化（Q4 → Q5）
- 调整温度参数
- 检查 prompt 设计

### Q4: 如何选择模型？
**A**:
- 通用任务：Llama 3
- 中文任务：Qwen
- 代码任务：DeepSeek
- 快速响应：Mistral

## 📚 学习资源

### 官方文档
- [Ollama 文档](https://github.com/ollama/ollama)
- [vLLM 文档](https://docs.vllm.ai/)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)

### 推荐阅读
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)

### 视频教程
- [Ollama 快速入门](https://www.youtube.com/watch?v=...)
- [vLLM 性能优化](https://www.youtube.com/watch?v=...)

## 🎓 学习检查清单

### 基础技能
- [ ] 能够使用 Ollama 部署模型
- [ ] 理解量化的原理和权衡
- [ ] 能够测试和对比不同配置
- [ ] 能够监控性能指标

### 进阶技能
- [ ] 能够使用 vLLM 优化性能
- [ ] 能够使用 LoRA 微调模型
- [ ] 能够实现推理优化
- [ ] 能够进行成本效益分析

### 高级技能
- [ ] 能够设计生产级部署方案
- [ ] 能够实现完整的监控和日志
- [ ] 能够处理各种部署问题
- [ ] 能够优化成本和性能

## 🚀 下一步

完成本阶段后，你应该能够：
- ✅ 独立部署和优化本地模型
- ✅ 根据需求选择合适的模型和配置
- ✅ 进行模型微调和评估
- ✅ 实现生产级的推理服务
- ✅ 进行成本效益分析和决策

**继续学习**：[第六阶段：深度学习框架](../06_深度学习框架/README.md)

---

**最后更新**：2026-02-07
