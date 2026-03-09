# 模型文件目录

此目录用于存放下载的模型文件。

**注意**：模型文件不会被提交到 Git 仓库。

## 目录结构

```
models/
├── llama3-8b/          # Llama 3 8B 模型
├── mistral-7b/         # Mistral 7B 模型
└── custom/             # 自定义微调模型
```

## 使用说明

1. 使用 Ollama 时，模型会自动下载到 Ollama 的默认目录
2. 使用 Hugging Face 模型时，可以设置 `HF_HOME` 环境变量指向此目录
3. 自定义微调的模型可以保存在 `custom/` 子目录中

## 环境变量

```bash
# 设置 Hugging Face 模型缓存目录
export HF_HOME=./models

# 设置 Transformers 缓存目录
export TRANSFORMERS_CACHE=./models
```
