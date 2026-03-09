# 01 - 基础API调用示例

## 项目目标
学习如何直接调用AI模型API，理解最基础的交互方式。

## 学习内容
- ✅ 环境配置和依赖安装
- ✅ API密钥管理
- ✅ 基础的API调用
- ✅ Prompt engineering入门
- ✅ Token计算和成本控制

## 文件说明

### 1. `requirements.txt`
项目依赖包列表

### 2. `.env.example`
环境变量模板文件（需要复制为`.env`并填入真实API密钥）

### 3. `01_openai_basic.py`
OpenAI API基础调用示例

### 4. `02_anthropic_basic.py`
Anthropic Claude API基础调用示例

### 5. `03_simple_chatbot.py`
简单的命令行聊天机器人

### 6. `04_prompt_engineering.py`
Prompt工程技巧示例

### 7. `05_token_counter.py`
Token计算和成本估算工具

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置API密钥
复制 `.env.example` 为 `.env`，并填入你的API密钥：
```bash
cp .env.example .env
```

编辑 `.env` 文件：
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### 3. 运行示例
```bash
# OpenAI基础示例
python 01_openai_basic.py

# Anthropic基础示例
python 02_anthropic_basic.py

# 命令行聊天机器人
python 03_simple_chatbot.py

# Prompt工程示例
python 04_prompt_engineering.py

# Token计算器
python 05_token_counter.py
```

## 学习笔记

### API调用的核心概念
1. **API密钥**：身份认证凭证
2. **模型选择**：不同模型有不同能力和价格
3. **Prompt**：给模型的指令
4. **Temperature**：控制输出的随机性（0-2）
5. **Max Tokens**：限制输出长度
6. **System Message**：设定模型的角色和行为

### 常用模型对比

#### OpenAI
- `gpt-4o`: 最新多模态模型，性能强
- `gpt-4o-mini`: 轻量版，便宜快速
- `gpt-3.5-turbo`: 经典模型，性价比高

#### Anthropic
- `claude-3-5-sonnet-20241022`: 最新Sonnet版本
- `claude-3-opus-20240229`: 最强模型
- `claude-3-haiku-20240307`: 最快最便宜

### Token和成本
- 1 token ≈ 0.75个英文单词
- 1 token ≈ 1.5-2个中文字符
- 成本 = (输入tokens × 输入价格 + 输出tokens × 输出价格) / 1M

### 最佳实践
1. 使用环境变量管理API密钥
2. 设置合理的max_tokens避免超支
3. 使用try-except处理API错误
4. 实现重试机制应对网络问题
5. 记录API调用日志便于调试

## 下一步
完成本示例后，可以进入 `02_Agent框架入门` 学习LangChain等框架。

---
**创建日期**：2026-01-24
**状态**：进行中
