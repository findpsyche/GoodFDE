# 模型文件目录

此目录用于存放下载的预训练模型和训练产出的模型文件。

## 注意事项

- 模型文件通常很大（数百MB到数GB），不应提交到 Git
- 使用 Hugging Face Hub 下载模型时，建议设置 `HF_HOME` 环境变量指向此目录
- 训练产出的 checkpoint 也会保存在此目录

## 常用模型

| 模型 | 大小 | 用途 |
|------|------|------|
| bert-base-uncased | ~440MB | 文本分类、NER |
| gpt2 | ~548MB | 文本生成 |
| t5-small | ~242MB | Seq2Seq 任务 |
| distilbert-base-uncased | ~268MB | 轻量级分类 |
