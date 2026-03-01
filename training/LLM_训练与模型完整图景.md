# LLM 训练与模型完整图景
> 面试补盲区文档 · 清华水利AI平台研发岗
> 目标：面对任何LLM训练、微调、部署、模型对比追问都能接住

---

## 一、Transformer 核心原理

### 1.1 Self-Attention 机制

```
第一性原理：Attention解决什么问题？
  RNN/LSTM：顺序处理，长距离依赖衰减
  Attention：任意两个token直接交互，并行计算

Self-Attention公式：
  Q = X · W_Q    (Query：我要查询什么)
  K = X · W_K    (Key：我有什么信息)
  V = X · W_V    (Value：我的信息内容)
  
  Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V
  
  √d_k：缩放因子，防止softmax梯度消失
  softmax：归一化为概率分布
  结果：每个token的表示 = 所有token的加权和
```

**直观理解：**
```
输入："我爱北京天安门"
Query("爱") 和所有Key计算相似度：
  "我"(0.3) "爱"(0.1) "北京"(0.4) "天安门"(0.2)
加权求和Value：
  "爱"的新表示 = 0.3*V("我") + 0.1*V("爱") + 0.4*V("北京") + 0.2*V("天安门")
→ "爱"的表示融合了上下文信息
```

### 1.2 Multi-Head Attention

```
为什么需要多头？
  单头Attention只能学到一种关系（如主谓关系）
  多头可以学到多种关系（主谓、动宾、修饰等）

实现：
  8个头 → 8组(W_Q, W_K, W_V) → 8个Attention结果 → Concat → 线性变换
  
  每个头的维度：d_model / num_heads
  例：768维模型，12个头 → 每个头64维
```

### 1.3 位置编码

```
问题：Attention没有位置信息（"我爱你"和"你爱我"的Attention结果相同）

解决方案1：绝对位置编码（原始Transformer）
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  优点：固定公式，不需要学习
  缺点：外推性差（训练512长度，推理1024会失效）

解决方案2：相对位置编码
  不编码绝对位置，编码token间的相对距离
  代表：T5的相对位置bias

解决方案3：RoPE（旋转位置编码）— 现代LLM主流
  用旋转矩阵编码位置信息
  优点：外推性好、计算高效
  代表：LLaMA、Qwen、GLM系列

解决方案4：ALiBi（Attention with Linear Biases）
  在Attention score上加线性偏置
  优点：外推性最好、无需额外参数
  代表：BLOOM、MPT
```

### 1.4 Transformer完整架构

```
Encoder-Decoder架构（原始Transformer）：
  Encoder: Self-Attention → FFN → 重复N层
  Decoder: Masked Self-Attention → Cross-Attention → FFN → 重复N层
  适用：机器翻译（输入输出都是序列）

Decoder-only架构（GPT系列）：
  只有Decoder，用Causal Mask（下三角矩阵）
  每个token只能看到之前的token
  适用：文本生成（自回归）
  代表：GPT、LLaMA、Qwen

Encoder-only架构（BERT系列）：
  只有Encoder，双向Attention
  适用：文本理解（分类、NER、问答）
  代表：BERT、RoBERTa
```

---

## 二、架构变体与优化

### 2.1 三大架构对比

| 维度        | Encoder-only | Decoder-only    | Encoder-Decoder     |
| --------- | ------------ | --------------- | ------------------- |
| 代表        | BERT、RoBERTa | GPT、LLaMA、Qwen  | T5、BART             |
| Attention | 双向（看全文）      | 单向（Causal Mask） | Encoder双向+Decoder单向 |
| 预训练任务     | MLM（掩码预测）    | CLM（下一个词预测）     | Span Corruption     |
| 适用        | 理解任务（分类/NER） | 生成任务（对话/写作）     | Seq2Seq（翻译/摘要）      |
| 推理效率      | 高（并行）        | 低（自回归）          | 中等                  |
| 当前趋势      | 逐渐被Decoder替代 | 主流（统一架构）        | 特定场景                |

**为什么Decoder-only成为主流？**
```
1. 统一架构：生成任务天然包含理解（生成需要理解输入）
2. 扩展性好：参数越大效果越好（Scaling Law）
3. 指令微调：通过SFT可以做任何任务
4. 工程简单：只需要一种架构，不需要Encoder+Decoder
```

### 2.2 关键优化技术

**FlashAttention（推理加速）**
```
问题：标准Attention需要O(N²)显存存储注意力矩阵
解决：分块计算，只在SRAM中存储中间结果
效果：显存占用降低，速度提升2-4倍
代表：FlashAttention-2（2023）
```

**GQA（Grouped-Query Attention）**
```
问题：MHA（Multi-Head Attention）每个头都有独立的K、V，显存占用大
解决：多个Query头共享一组K、V
  MHA: 32个头 → 32组(K,V)
  GQA: 32个头 → 8组(K,V)（4个头共享1组）
  MQA: 32个头 → 1组(K,V)（极端情况）
效果：推理速度提升，显存占用降低
代表：LLaMA-2、Mistral
```

**MoE（Mixture of Experts）**
```
原理：不是所有参数都参与每次计算
  模型有N个Expert（FFN层）
  Router网络选择Top-K个Expert激活
  其他Expert不参与计算
效果：
  参数量大（如1.8T），但激活参数少（如47B）
  推理成本接近小模型，能力接近大模型
代表：Mixtral 8x7B、DeepSeek-V2
```

**KV-Cache（推理优化）**
```
问题：自回归生成时，每次都要重新计算之前token的K、V
解决：缓存已计算的K、V，新token只计算自己的K、V
效果：推理速度提升10倍+
代价：显存占用增加（需要存储KV-Cache）
  KV-Cache大小 = 2 × num_layers × hidden_size × seq_len × batch_size × 2字节
  例：LLaMA-7B，2048长度，batch=1 → 约2GB显存
```

---

## 三、Pre-training（预训练）

### 3.1 预训练任务

```
Causal Language Modeling (CLM) — Decoder-only主流
  任务：预测下一个token
  输入："我爱北京天"
  目标：预测"安"
  Loss：交叉熵损失 -log P(安 | 我爱北京天)

Masked Language Modeling (MLM) — Encoder-only
  任务：预测被掩码的token
  输入："我爱[MASK]京天安门"
  目标：预测"北"
  代表：BERT

Span Corruption — Encoder-Decoder
  任务：预测被掩盖的span
  输入："我爱<X>天安门"
  目标：生成"<X>北京"
  代表：T5
```

### 3.2 预训练数据

```
数据来源：
  - 网页爬取：Common Crawl（最大）
  - 书籍：Books3、Gutenberg
  - 代码：GitHub、StackOverflow
  - 学术：arXiv、PubMed
  - 对话：Reddit、论坛

数据清洗（关键）：
  1. 去重：MinHash、SimHash
  2. 质量过滤：
     - 语言检测（只保留目标语言）
     - 长度过滤（太短/太长的文本）
     - 困惑度过滤（用小模型打分，去掉低质量文本）
  3. 有害内容过滤：
     - 敏感词过滤
     - 分类器过滤（暴力/色情/仇恨言论）
  4. PII脱敏：去除邮箱、电话、身份证号

数据配比（重要）：
  不同来源数据的混合比例影响模型能力
  例：LLaMA的配比
    - 网页：67%
    - 书籍：15%
    - 代码：4.5%
    - 学术：4.5%
    - 其他：9%
```

### 3.3 Scaling Laws

```
Kaplan et al. (2020) 发现的规律：
  Loss ∝ N^(-α)
  N：模型参数量、数据量、计算量
  α：幂律指数

关键结论：
  1. 模型越大，Loss越低（但边际收益递减）
  2. 数据越多，Loss越低
  3. 计算量越大，Loss越低
  4. 三者需要平衡：
     - 参数量翻倍 → 数据量也要翻倍
     - 否则会过拟合或欠拟合

Chinchilla Scaling Law (2022)：
  最优配比：参数量 : 数据量 = 1 : 20
  例：70B模型 → 需要1.4T tokens数据
  → LLaMA、Qwen等都遵循这个规律
```

### 3.4 分布式训练

```
数据并行（Data Parallelism）：
  每个GPU一份完整模型，不同数据
  梯度同步：AllReduce
  适用：模型能放进单卡

模型并行（Model Parallelism）：
  模型切分到多个GPU
  - 张量并行（Tensor Parallelism）：切分单层
  - 流水线并行（Pipeline Parallelism）：切分多层
  适用：模型太大，单卡放不下

ZeRO（Zero Redundancy Optimizer）：
  DeepSpeed的核心技术
  ZeRO-1：切分优化器状态
  ZeRO-2：切分梯度
  ZeRO-3：切分模型参数
  效果：显存占用降低N倍（N=GPU数）

混合精度训练：
  FP32（全精度）→ FP16/BF16（半精度）
  效果：速度提升2倍，显存减半
  注意：需要Loss Scaling防止梯度下溢
```

---

## 四、SFT（Supervised Fine-Tuning）

### 4.1 SFT原理

```
目标：让预训练模型学会"对话"和"遵循指令"

数据格式：
  {
    "instruction": "解释什么是机器学习",
    "input": "",
    "output": "机器学习是一种人工智能技术..."
  }

训练方式：
  和预训练一样，预测下一个token
  但只计算output部分的Loss（instruction部分不计算）
  → 模型学会"看到instruction，生成output"

关键：数据质量 > 数据数量
  1000条高质量数据 > 10万条低质量数据
```

### 4.2 指令数据构造

```
方法1：人工标注
  优点：质量最高
  缺点：成本高、规模小
  适用：种子数据

方法2：Self-Instruct（自举）
  用GPT-4生成指令数据
  流程：
    1. 人工写175个种子任务
    2. GPT-4生成新任务
    3. GPT-4生成任务的输入输出
    4. 过滤低质量数据
  代表：Alpaca（52K数据）

方法3：Evol-Instruct（进化）
  让GPT-4把简单指令"进化"成复杂指令
  进化方向：
    - 增加约束
    - 深化难度
    - 增加推理步骤
    - 复杂化输入
  代表：WizardLM

方法4：从真实用户数据提取
  收集用户和ChatGPT的对话
  过滤、清洗、去重
  代表：ShareGPT
```

### 4.3 SFT训练技巧

```
超参数：
  学习率：2e-5（比预训练小10倍）
  Batch size：128-256
  Epochs：3-5（不要太多，容易过拟合）
  Warmup：10%步数

数据增强：
  - 同义改写：用GPT-4改写instruction
  - 多样化output：同一instruction生成多个output
  - 负样本：错误的output（用于对比学习）

评估：
  - 自动评估：ROUGE、BLEU（不准确）
  - LLM-as-Judge：用GPT-4评分
  - 人工评估：最准确但成本高
```

---

## 五、RLHF（人类反馈强化学习）

### 5.1 RLHF 完整流程

```
三阶段流程：

阶段1：SFT（已讲）
  预训练模型 → SFT数据微调 → SFT模型

阶段2：训练Reward Model（奖励模型）
  数据：同一个prompt，生成多个response，人工排序
    prompt: "解释量子力学"
    response_A: "量子力学是研究微观粒子..."（好）
    response_B: "量子力学就是薛定谔的猫..."（一般）
    response_C: "我不知道"（差）
    人工标注：A > B > C

  训练：
    RM(prompt, response_A) > RM(prompt, response_B) > RM(prompt, response_C)
    Loss = -log(sigmoid(RM(A) - RM(B)))  # Bradley-Terry模型

阶段3：PPO优化（强化学习）
  目标：最大化Reward，同时不偏离SFT模型太远
  
  Loss = E[RM(prompt, response)] - β * KL(π_RL || π_SFT)
  
  π_RL：当前RL策略（正在训练的模型）
  π_SFT：SFT模型（参考模型）
  β：KL惩罚系数（防止模型"hack"奖励函数）
  
  PPO算法：
    1. 用当前模型生成response
    2. Reward Model打分
    3. 计算优势函数（Advantage）
    4. 更新模型参数（clip防止更新过大）
```

### 5.2 RLHF 的问题

```
1. 训练不稳定：PPO超参数敏感，容易崩溃
2. Reward Hacking：模型学会"讨好"RM而不是真正变好
   例：生成冗长但空洞的回答（RM给高分因为"看起来详细"）
3. 成本高：需要人工标注偏好数据
4. 对齐税（Alignment Tax）：RLHF后模型在某些任务上性能下降
```

---

## 六、DPO（Direct Preference Optimization）

### 6.1 DPO vs RLHF

```
RLHF的问题：需要训练RM + PPO，流程复杂
DPO的解法：直接从偏好数据优化模型，跳过RM和PPO

DPO Loss：
  L = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
  
  y_w：偏好的response（winner）
  y_l：不偏好的response（loser）
  π：当前模型
  π_ref：参考模型（SFT模型）
  β：温度参数

直觉理解：
  增大偏好response的概率
  减小不偏好response的概率
  同时不偏离参考模型太远
```

### 6.2 DPO vs RLHF 对比

| 维度 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 流程 | SFT → RM → PPO（三阶段） | SFT → DPO（两阶段） |
| 需要RM | ✅ 需要训练 | ❌ 不需要 |
| 训练稳定性 | 差（PPO超参敏感） | 好（标准监督学习） |
| 计算成本 | 高（4个模型同时在显存） | 低（2个模型） |
| 效果 | 略好（理论上限高） | 接近RLHF |
| 实现复杂度 | 高 | 低 |
| 当前趋势 | 逐渐被DPO替代 | 主流选择 |

**面试一句话：** "DPO把RLHF的三阶段简化为两阶段，去掉了Reward Model和PPO，直接从偏好数据优化。训练更稳定、成本更低，效果接近RLHF。"

---

## 七、PEFT（参数高效微调）

### 7.1 为什么需要PEFT

```
全量微调的问题：
  LLaMA-7B：7B参数 × 4字节(FP32) = 28GB
  + 梯度：28GB
  + 优化器状态(Adam)：56GB
  总计：~112GB显存 → 需要多张A100

PEFT的解法：只训练少量参数，冻结大部分参数
  训练参数量：0.1% - 5%
  显存需求：大幅降低
```

### 7.2 LoRA（Low-Rank Adaptation）— 最主流

```
原理：
  原始权重矩阵 W ∈ R^(d×d)
  微调时不直接更新W，而是学习低秩分解：
  W' = W + ΔW = W + B·A
  
  A ∈ R^(d×r)  （降维）
  B ∈ R^(r×d)  （升维）
  r << d（秩，通常r=8或16）
  
  参数量对比：
    全量微调：d × d = 4096 × 4096 = 16M
    LoRA(r=8)：d × r + r × d = 4096 × 8 × 2 = 65K
    → 参数量减少250倍

代码示例：
  from peft import LoraConfig, get_peft_model
  
  config = LoraConfig(
      r=8,                    # 秩
      lora_alpha=16,          # 缩放因子
      target_modules=["q_proj", "v_proj"],  # 应用到哪些层
      lora_dropout=0.05,
      task_type="CAUSAL_LM"
  )
  model = get_peft_model(base_model, config)
  # 可训练参数：~0.1%

关键超参数：
  r（秩）：越大表达能力越强，但参数越多。通常8-64
  lora_alpha：缩放因子，通常设为2r
  target_modules：通常选q_proj和v_proj（Attention层）
```

### 7.3 QLoRA（量化LoRA）

```
原理：基础模型用4-bit量化，LoRA部分用FP16
  → 7B模型只需要~6GB显存（单张消费级GPU可跑）

技术细节：
  1. NF4量化：4-bit NormalFloat，比INT4精度更高
  2. 双重量化：量化参数本身也量化
  3. 分页优化器：显存不够时自动卸载到CPU

代码：
  from transformers import BitsAndBytesConfig
  
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True
  )
  model = AutoModelForCausalLM.from_pretrained(
      "meta-llama/Llama-2-7b-hf",
      quantization_config=bnb_config
  )
```

### 7.4 PEFT方法对比

| 方法            | 原理          | 参数量    | 效果     | 适用    |
| ------------- | ----------- | ------ | ------ | ----- |
| LoRA          | 低秩分解        | 0.1-1% | 接近全量   | 最主流   |
| QLoRA         | LoRA+4bit量化 | 0.1-1% | 接近LoRA | 显存受限  |
| Adapter       | 插入小型网络层     | 1-5%   | 好      | 多任务   |
| Prefix Tuning | 学习虚拟前缀token | <0.1%  | 中等     | 生成任务  |
| P-Tuning v2   | 每层加可学习前缀    | 0.1-1% | 好      | 中文NLU |
| 全量微调          | 更新所有参数      | 100%   | 最好     | 资源充足  |

---

## 八、推理优化

### 8.1 KV-Cache 详解

```
自回归生成过程：
  Step 1: "我" → 计算K1,V1 → 预测"爱"
  Step 2: "我爱" → 需要K1,V1,K2,V2 → 预测"北"
  Step 3: "我爱北" → 需要K1,V1,K2,V2,K3,V3 → 预测"京"

没有KV-Cache：每步都重新计算所有K,V → O(N²)
有KV-Cache：缓存之前的K,V，只计算新token → O(N)

显存占用：
  KV-Cache = 2 × num_layers × num_heads × head_dim × seq_len × batch × dtype_size
  LLaMA-7B, seq=2048, batch=1, FP16:
  = 2 × 32 × 32 × 128 × 2048 × 1 × 2 bytes ≈ 1GB
```

### 8.2 PagedAttention（vLLM核心）

```
问题：KV-Cache需要连续显存，不同请求的seq_len不同
  → 显存碎片化严重，利用率低

解决：借鉴OS虚拟内存的分页机制
  KV-Cache分成固定大小的Block（如16 tokens）
  Block可以不连续存储
  用Page Table管理映射关系

效果：
  显存利用率从50-60%提升到95%+
  支持更大的batch size → 吞吐量提升2-4倍
```

### 8.3 Continuous Batching

```
传统Batching：
  等一批请求到齐 → 一起推理 → 等最长的请求完成 → 返回所有结果
  问题：短请求等长请求，GPU利用率低

Continuous Batching：
  请求随到随处理
  某个请求完成 → 立即加入新请求
  不等其他请求完成
  
效果：吞吐量提升2-3倍
```

### 8.4 Speculative Decoding（投机解码）

```
原理：
  用小模型（Draft Model）快速生成K个候选token
  用大模型（Target Model）一次性验证K个token
  接受正确的，拒绝错误的

效果：
  大模型的推理速度提升2-3倍
  输出质量和大模型完全一致（数学上等价）

代表：Medusa、Eagle
```

---

## 九、模型量化

### 9.1 量化原理

```
量化 = 用低精度数值表示高精度数值

FP32（32位浮点）→ FP16（16位）→ INT8（8位）→ INT4（4位）

精度 vs 速度 vs 显存：
  FP32: 精度最高，速度最慢，显存最大
  FP16/BF16: 精度略降，速度2倍，显存减半
  INT8: 精度可接受，速度4倍，显存1/4
  INT4: 精度有损，速度8倍，显存1/8
```

### 9.2 量化方法对比

| 方法 | 精度 | 速度 | 显存 | 需要校准数据 | 适用 |
|------|------|------|------|------------|------|
| GPTQ | INT4 | 快 | 小 | ✅ 需要 | GPU推理 |
| AWQ | INT4 | 快 | 小 | ✅ 需要 | GPU推理（质量略优） |
| GGUF | INT4/5/8 | 中 | 小 | ❌ 不需要 | CPU推理（llama.cpp） |
| bitsandbytes | NF4/INT8 | 中 | 小 | ❌ 不需要 | QLoRA训练 |

```
GPTQ：
  逐层量化，用少量校准数据（128条）最小化量化误差
  适合GPU推理，和vLLM/TGI配合好

AWQ（Activation-aware Weight Quantization）：
  保护重要权重（对激活值影响大的权重用更高精度）
  质量略优于GPTQ

GGUF（GPT-Generated Unified Format）：
  llama.cpp的格式，支持CPU推理
  支持多种量化级别（Q4_K_M、Q5_K_M等）
  适合本地部署、边缘设备

bitsandbytes：
  HuggingFace集成，一行代码量化
  NF4：QLoRA专用，训练时量化
```

---

## 十、推理框架对比

| 维度 | vLLM | TGI | Ollama | llama.cpp |
|------|------|-----|--------|-----------|
| 开发者 | UC Berkeley | HuggingFace | Ollama | Georgi Gerganov |
| 核心技术 | PagedAttention | Continuous Batching | llama.cpp封装 | GGML/GGUF |
| 硬件 | GPU | GPU | GPU/CPU | CPU/GPU |
| API兼容 | OpenAI兼容 | 自有API | OpenAI兼容 | 自有API |
| 量化支持 | GPTQ/AWQ/FP16 | GPTQ/AWQ/BNB | GGUF | GGUF |
| 吞吐量 | 最高 | 高 | 低（单用户） | 低 |
| 适用 | 生产部署 | 生产部署 | 本地开发 | 边缘/嵌入式 |
| 部署复杂度 | 中 | 低（Docker） | 最低 | 低 |

**面试选型建议：**
```
生产环境高并发 → vLLM（PagedAttention吞吐最高）
HuggingFace生态 → TGI（集成最好）
本地开发测试 → Ollama（一行命令启动）
无GPU/边缘设备 → llama.cpp（纯CPU推理）
```

---

## 十一、显存计算与GPU选型

### 11.1 显存计算公式

```
模型加载显存：
  参数量 × 每参数字节数
  FP32: 7B × 4 = 28GB
  FP16: 7B × 2 = 14GB
  INT8: 7B × 1 = 7GB
  INT4: 7B × 0.5 = 3.5GB

训练显存（全量微调）：
  模型参数: P × 2 (FP16)
  梯度: P × 2
  优化器状态(Adam): P × 8 (FP32的momentum + variance)
  激活值: 取决于batch size和seq_len
  总计 ≈ P × 14 + 激活值
  
  7B模型全量微调 ≈ 7B × 14 ≈ 98GB → 需要2-4张A100(80GB)

LoRA训练显存：
  模型参数(冻结): P × 2 (FP16)
  LoRA参数: ~0.1% × P × 2
  LoRA梯度+优化器: ~0.1% × P × 12
  总计 ≈ P × 2 + 少量
  
  7B模型LoRA ≈ 14GB + 少量 → 1张A100(40GB)
  7B模型QLoRA ≈ 3.5GB + 少量 → 1张RTX 3090(24GB)
```

### 11.2 GPU选型

| GPU | 显存 | 价格(云) | 适用 |
|-----|------|---------|------|
| RTX 3090 | 24GB | 便宜 | QLoRA 7B、推理7B-INT4 |
| RTX 4090 | 24GB | 中等 | QLoRA 7B、推理13B-INT4 |
| A100 40GB | 40GB | 贵 | LoRA 7B、推理13B-FP16 |
| A100 80GB | 80GB | 很贵 | 全量微调7B、推理70B-INT4 |
| H100 80GB | 80GB | 最贵 | 全量微调13B、推理70B-FP16 |
| 8×A100 80GB | 640GB | 极贵 | 预训练、全量微调70B |

---

## 十二、闭源模型对比

| 模型 | 开发者 | 上下文 | 特点 | 适用 |
|------|--------|--------|------|------|
| GPT-4o | OpenAI | 128K | 多模态、速度快 | 通用最强 |
| GPT-4 Turbo | OpenAI | 128K | 推理能力强 | 复杂推理 |
| Claude 3.5 Sonnet | Anthropic | 200K | 代码能力强、长上下文 | 代码生成 |
| Claude 3 Opus | Anthropic | 200K | 最强推理 | 复杂任务 |
| Gemini 1.5 Pro | Google | 2M | 超长上下文 | 长文档分析 |
| Gemini 1.5 Flash | Google | 1M | 速度快、便宜 | 高频调用 |

**面试关键点：**
```
GPT-4o：综合能力最强，多模态支持好，API稳定
Claude 3.5 Sonnet：代码能力最强，Tool Calling质量高，200K上下文
Gemini 1.5 Pro：2M上下文独一无二，适合处理超长文档
选型：看具体任务需求和预算
```

---

## 十三、开源模型对比

### 13.1 主流开源模型

| 系列 | 开发者 | 参数规模 | 特点 | 许可 |
|------|--------|---------|------|------|
| LLaMA 3 | Meta | 8B/70B/405B | 基座模型标杆 | LLaMA 3 License |
| Qwen 2.5 | 阿里 | 0.5B-72B | 中文最强、多语言 | Apache 2.0 |
| DeepSeek V2 | DeepSeek | 236B(21B激活) | MoE架构、性价比高 | MIT |
| Mistral | Mistral AI | 7B/8x7B/8x22B | 欧洲最强、MoE | Apache 2.0 |
| Yi | 零一万物 | 6B/34B | 中文好、长上下文 | Apache 2.0 |
| ChatGLM | 智谱AI | 6B/32B | 中文对话 | ChatGLM License |
| InternLM | 上海AI Lab | 7B/20B | 中文理解强 | Apache 2.0 |
| Baichuan | 百川智能 | 7B/13B | 中文垂直领域 | Baichuan License |

### 13.2 模型能力维度对比

```
推理能力（数学、逻辑）：
  Tier 1: GPT-4o, Claude 3 Opus, Gemini 1.5 Pro
  Tier 2: LLaMA 3 70B, Qwen 2.5 72B, DeepSeek V2
  Tier 3: Mistral 7B, Qwen 2.5 7B

代码能力：
  Tier 1: Claude 3.5 Sonnet, GPT-4o
  Tier 2: DeepSeek Coder, Qwen 2.5 Coder
  Tier 3: LLaMA 3 70B, Mistral 7B

中文能力：
  Tier 1: Qwen 2.5, ChatGLM, InternLM
  Tier 2: Yi, Baichuan
  Tier 3: LLaMA 3（中文弱）

Tool Calling：
  Tier 1: Claude 3.5, GPT-4o
  Tier 2: Qwen 2.5, Mistral
  Tier 3: LLaMA 3（需要微调）

长上下文：
  Tier 1: Gemini 1.5 Pro (2M), Claude 3 (200K)
  Tier 2: Qwen 2.5 (128K), Yi (200K)
  Tier 3: LLaMA 3 (128K), Mistral (32K)

多模态：
  Tier 1: GPT-4o, Gemini 1.5 Pro
  Tier 2: Qwen-VL, LLaVA
  Tier 3: 大部分开源模型不支持
```

### 13.3 开源模型选型建议

```
通用对话（中文）：
  首选：Qwen 2.5 7B/14B（中文最强、Apache 2.0）
  备选：ChatGLM 32B（对话优化好）

代码生成：
  首选：DeepSeek Coder 33B（代码专用）
  备选：Qwen 2.5 Coder 7B（轻量级）

英文任务：
  首选：LLaMA 3 8B/70B（基座最强）
  备选：Mistral 7B（欧洲标杆）

资源受限：
  首选：Qwen 2.5 0.5B/1.5B（手机可跑）
  备选：Phi-3 Mini（微软出品）

商业化：
  首选：Apache 2.0许可（Qwen、Mistral、Yi）
  避免：限制商用的许可（LLaMA 3需审查）
```

---

## 十四、评测体系

### 14.1 通用能力评测

| 基准 | 内容 | GPT-4o | Claude 3 Opus | LLaMA 3 70B | Qwen 2.5 72B |
|------|------|--------|---------------|-------------|--------------|
| MMLU | 57学科选择题 | 86.4% | 86.8% | 82.0% | 85.3% |
| C-Eval | 52学科中文 | 69.9% | 68.4% | 60.1% | 91.6% |
| HumanEval | Python代码 | 67.0% | 92.0% | 81.7% | 65.9% |
| MBPP | 基础编程 | 80.0% | 87.0% | 82.6% | 80.2% |

### 14.2 RAG评测

```
RAGAS框架：
  Faithfulness（忠实度）：回答是否忠实于检索文档
  Answer Relevance（相关性）：回答是否回答了问题
  Context Recall（召回率）：检索到的文档是否包含答案
  Context Precision（精确率）：检索到的文档是否相关

自定义评测：
  Recall@K：Top-K中包含正确文档的比例
  MRR：第一个正确文档的排名倒数
  NDCG@K：考虑排名的综合指标
```

---

## 十五、LLM监控指标

### 15.1 性能指标

```
延迟（Latency）：
  TTFT（Time To First Token）：首token延迟
  TPOT（Time Per Output Token）：每token延迟
  E2E Latency：端到端延迟

吞吐量（Throughput）：
  QPS（Queries Per Second）：每秒请求数
  TPS（Tokens Per Second）：每秒生成token数

资源利用率：
  GPU利用率：目标>80%
  显存占用：监控KV-Cache增长
  CPU利用率：监控tokenizer瓶颈
```

### 15.2 质量指标

```
输出质量：
  平均输出长度：监控是否异常（过长/过短）
  拒答率：模型说"我不知道"的比例
  幻觉率：LLM-as-Judge评估

用户反馈：
  点赞率 / 点踩率 / 重试率

成本指标：
  Token消耗：输入+输出token总数
  API成本：按token计费的总成本
  单次对话成本：平均每次对话的成本
```

### 15.3 监控工具

```
Langfuse：LLM可观测性平台（开源，可私有化）
LangSmith：LangChain官方监控
Phoenix（Arize AI）：实时监控+幻觉检测
Weights & Biases：实验追踪
```

---

## 十六、20个高频面试问题

### 16.1 基础原理

**Q1: Self-Attention和RNN的本质区别？**
```
RNN：顺序处理，t时刻依赖t-1，无法并行，长距离依赖衰减
Attention：任意两个token直接交互，完全并行，长距离不衰减
代价：Attention是O(N^2)复杂度，RNN是O(N)
```

**Q2: 为什么需要位置编码？**
```
Attention没有位置信息，"我爱你"和"你爱我"结果相同。
位置编码给每个token加上位置信息。
现代LLM主流用RoPE（旋转位置编码），外推性好。
```

**Q3: Multi-Head Attention为什么比Single-Head好？**
```
单头只能学到一种关系（如主谓关系）
多头可以学到多种关系（主谓、动宾、修饰等）
每个头关注不同的语义角度，最后concat融合。
```

**Q4: Decoder-only为什么成为主流？**
```
1. 统一架构：生成任务天然包含理解
2. Scaling Law：参数越大效果越好
3. 指令微调：通过SFT可以做任何任务
4. 工程简单：只需要一种架构
```

### 16.2 训练相关

**Q5: 预训练和微调有什么区别？**
```
预训练：海量无标注文本，预测下一个token，学习通用表示，成本极高
微调：少量标注数据，特定任务，适配下游，成本低
```

**Q6: SFT和RLHF的区别？**
```
SFT：监督学习，学习"正确答案"
  数据：(instruction, output)对
RLHF：强化学习，学习"人类偏好"
  数据：(prompt, response_A > response_B)
  效果：输出更符合人类偏好（有用、无害、诚实）
```

**Q7: DPO为什么比RLHF好？**
```
RLHF：需要训练RM + PPO，流程复杂，训练不稳定
DPO：直接从偏好数据优化，跳过RM和PPO
  训练稳定、成本低、效果接近RLHF
当前趋势：DPO逐渐替代RLHF
```

**Q8: LoRA的原理是什么？**
```
原始权重W不动，学习低秩分解 W_new = W + B*A
A: d*r（降维），B: r*d（升维），r<<d
参数量减少250倍，显存大幅降低
优点：可插拔、训练快、效果接近全量微调
```

### 16.3 推理优化

**Q9: KV-Cache是什么？**
```
自回归生成时缓存已计算的K、V，新token只计算自己的
效果：推理速度提升10倍+
代价：显存占用增加
```

**Q10: vLLM的PagedAttention是什么？**
```
借鉴OS虚拟内存，KV-Cache分页存储，Block可以不连续
效果：显存利用率从50%提升到95%+，吞吐量提升2-4倍
```

**Q11: 模型量化有哪些方法？**
```
GPTQ/AWQ：GPU推理，需要校准数据，INT4
GGUF：CPU推理（llama.cpp），不需要校准数据
bitsandbytes：QLoRA训练专用，NF4
```

### 16.4 模型对比

**Q12: GPT-4o和Claude 3.5 Sonnet有什么区别？**
```
GPT-4o：综合最强，多模态好，API稳定
Claude 3.5 Sonnet：代码最强（HumanEval 92%），Tool Calling质量高，200K上下文
选型：代码任务选Claude，其他选GPT-4o
```

**Q13: 开源模型怎么选？**
```
中文：Qwen 2.5（最强、Apache 2.0）
代码：DeepSeek Coder
英文：LLaMA 3
资源受限：Qwen 2.5 0.5B
商业化：选Apache 2.0许可
```

**Q14: LLaMA 3和Qwen 2.5有什么区别？**
```
LLaMA 3：英文最强，中文弱，许可限制商用
Qwen 2.5：中文最强，多语言好，Apache 2.0
选型：英文选LLaMA，中文选Qwen
```

### 16.5 实战问题

**Q15: 如何评估LLM的效果？**
```
自动评估：MMLU、C-Eval、HumanEval
LLM-as-Judge：用GPT-4评分
人工评估：A/B测试
RAG评估：Recall@K、MRR、Faithfulness
```

**Q16: 如何降低LLM推理成本？**
```
1. 小模型做简单任务
2. INT4量化
3. 语义缓存
4. Prompt精简
5. 批处理
6. 开源模型自部署
```

**Q17: 如何提升RAG效果？**
```
检索：两阶段检索 + 混合检索 + Query改写
生成：Prompt工程 + 上下文压缩 + 引用来源
评估：Recall@K + Faithfulness + A/B测试
```

**Q18: 如何部署LLM到生产环境？**
```
框架：vLLM（高并发）/ TGI / Ollama（本地）
量化：GPTQ/AWQ（GPU）/ GGUF（CPU）
监控：TTFT、TPOT、QPS、幻觉率、Token消耗
容灾：多模型备份 + 降级策略 + 限流
```

**Q19: 如何微调一个7B模型？**
```
数据：1000-10000条高质量(instruction, output)
方法：QLoRA（4-bit + LoRA）
硬件：1张RTX 3090（24GB）
超参：lr=2e-5, batch=4, epochs=3
部署：合并LoRA → 量化 → vLLM
```

**Q20: 如何处理LLM的幻觉问题？**
```
预防：RAG注入事实 + Prompt约束 + 低temperature
检测：LLM-as-Judge + 事实核查 + 置信度
修正：重新生成 + 人工审核 + RLHF/DPO微调
```

---

## 十七、总结：核心能力地图

```
Transformer原理：
  Self-Attention / Multi-Head / RoPE / 三大架构

训练流程：
  Pre-training(CLM) / SFT / RLHF vs DPO / LoRA+QLoRA

推理优化：
  KV-Cache / PagedAttention / Continuous Batching / Speculative Decoding

量化与部署：
  GPTQ/AWQ/GGUF / vLLM/TGI/Ollama / 显存计算 / GPU选型

模型对比：
  闭源(GPT-4o/Claude/Gemini) / 开源(LLaMA/Qwen/DeepSeek)

评测与监控：
  MMLU/HumanEval/C-Eval / RAGAS / TTFT/TPOT/QPS
```

**面试前最后检查：**
- [ ] 能画出Transformer架构图
- [ ] 能解释LoRA的原理和公式
- [ ] 能说出3种推理优化技术
- [ ] 能对比5个以上的开源模型
- [ ] 能回答"如何降低推理成本"

---

**文档完成时间：** 2026-02-28
**适用岗位：** 清华水利AI平台研发（Python工程 + RAG/Agent + Web全栈）
**配合文档：**
- Web2_全栈开发完整图景.md（前后端+中间件+DevOps）
- 清华岗位_完整冲刺手册_含Challenger拷问.md（项目深挖+LLM应用）
- Part5_Challenger拷问手册_独立版.md（技术决策追问）
- 面试知识清单_LLM_Web系统开发.md（LLM x Web基础）
