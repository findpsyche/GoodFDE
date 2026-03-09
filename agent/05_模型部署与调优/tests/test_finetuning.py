"""
测试微调功能
"""

import pytest
import torch

# 检查依赖是否可用
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="Dependencies not installed")
class TestFineTuning:
    """微调测试"""

    @pytest.fixture
    def small_model(self):
        """小模型 fixture"""
        model_name = "facebook/opt-125m"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        return {"model": model, "tokenizer": tokenizer}

    def test_lora_config(self):
        """测试 LoRA 配置"""
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        assert lora_config.r == 8
        assert lora_config.lora_alpha == 16
        assert "q_proj" in lora_config.target_modules

    def test_apply_lora(self, small_model):
        """测试应用 LoRA"""
        model = small_model["model"]

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        peft_model = get_peft_model(model, lora_config)

        # 验证可训练参数
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())

        assert trainable_params > 0
        assert trainable_params < total_params

    def test_data_preparation(self, small_model):
        """测试数据准备"""
        tokenizer = small_model["tokenizer"]

        # 示例数据
        data = [
            {"text": "This is a test sentence."},
            {"text": "Another test sentence."}
        ]

        dataset = Dataset.from_list(data)

        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=128,
                padding="max_length"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        assert len(tokenized_dataset) == len(data)
        assert "input_ids" in tokenized_dataset[0]
        assert "attention_mask" in tokenized_dataset[0]

    def test_model_inference(self, small_model):
        """测试模型推理"""
        model = small_model["model"]
        tokenizer = small_model["tokenizer"]

        prompt = "Hello, world!"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        assert len(generated_text) > len(prompt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
