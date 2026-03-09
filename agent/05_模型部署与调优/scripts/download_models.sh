#!/bin/bash

# 模型下载脚本
# 用于批量下载 Ollama 模型

set -e

echo "🚀 Ollama 模型下载脚本"
echo "======================="
echo ""

# 检查 Ollama 是否安装
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama 未安装"
    echo "请访问 https://ollama.com 安装 Ollama"
    exit 1
fi

echo "✅ Ollama 已安装"
echo ""

# 推荐的模型列表
declare -A MODELS=(
    ["llama3:8b"]="Llama 3 8B (推荐)"
    ["llama3:8b-q4_K_M"]="Llama 3 8B Q4 量化"
    ["llama3:8b-q5_K_M"]="Llama 3 8B Q5 量化"
    ["mistral:7b"]="Mistral 7B"
    ["qwen:7b"]="Qwen 7B (中文优化)"
    ["phi3:mini"]="Phi-3 Mini (3.8B)"
)

echo "📋 可用模型:"
echo ""

i=1
model_keys=()
for model in "${!MODELS[@]}"; do
    echo "  $i. $model - ${MODELS[$model]}"
    model_keys+=("$model")
    ((i++))
done

echo ""
echo "  0. 下载所有模型"
echo "  q. 退出"
echo ""

read -p "请选择要下载的模型 (输入数字): " choice

if [ "$choice" = "q" ]; then
    echo "👋 退出"
    exit 0
fi

if [ "$choice" = "0" ]; then
    echo ""
    echo "📥 开始下载所有模型..."
    echo ""

    for model in "${model_keys[@]}"; do
        echo "⬇️  下载: $model"
        ollama pull "$model"
        echo ""
    done

    echo "✅ 所有模型下载完成！"
else
    if [ "$choice" -ge 1 ] && [ "$choice" -le "${#model_keys[@]}" ]; then
        selected_model="${model_keys[$((choice-1))]}"
        echo ""
        echo "📥 下载: $selected_model"
        ollama pull "$selected_model"
        echo ""
        echo "✅ 下载完成！"
    else
        echo "❌ 无效选项"
        exit 1
    fi
fi

echo ""
echo "📊 已安装的模型:"
ollama list
