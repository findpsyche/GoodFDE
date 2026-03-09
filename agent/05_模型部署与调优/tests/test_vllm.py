"""
测试 vLLM 部署
"""

import pytest

# 检查 vLLM 是否可用
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
class TestVLLM:
    """vLLM 测试"""

    @pytest.fixture
    def vllm_model(self):
        """vLLM 模型 fixture"""
        # 使用小模型进行测试
        model = LLM(
            model="facebook/opt-125m",
            max_model_len=512,
            gpu_memory_utilization=0.5
        )
        return model

    def test_basic_generation(self, vllm_model):
        """测试基本生成"""
        prompts = ["Hello, my name is"]

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=20
        )

        outputs = vllm_model.generate(prompts, sampling_params)

        assert len(outputs) == 1
        assert len(outputs[0].outputs) > 0
        assert len(outputs[0].outputs[0].text) > 0

    def test_batch_generation(self, vllm_model):
        """测试批量生成"""
        prompts = [
            "Hello, my name is",
            "The capital of France is",
            "Python is a"
        ]

        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=10
        )

        outputs = vllm_model.generate(prompts, sampling_params)

        assert len(outputs) == len(prompts)

        for output in outputs:
            assert len(output.outputs) > 0
            assert len(output.outputs[0].text) > 0

    def test_different_temperatures(self, vllm_model):
        """测试不同温度参数"""
        prompt = ["Tell me a story."]

        temperatures = [0.1, 0.7, 1.5]
        outputs_list = []

        for temp in temperatures:
            sampling_params = SamplingParams(
                temperature=temp,
                max_tokens=50
            )

            outputs = vllm_model.generate(prompt, sampling_params)
            outputs_list.append(outputs[0].outputs[0].text)

        # 验证不同温度产生不同输出
        assert len(set(outputs_list)) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
