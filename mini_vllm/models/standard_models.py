# mini_vllm/models/standard_models.py
# Standard model families that use the base ForCausalLM with config defaults.
from mini_vllm.models.model import ForCausalLM


class LlamaForCausalLM(ForCausalLM):
    """LLaMA / LLaMA-2 / LLaMA-3 family."""
    def __init__(self, config: dict):
        config.setdefault("attention_bias", False)
        config.setdefault("hidden_act", "silu")
        config.setdefault("rope_theta", 10000.0)
        super().__init__(config)


class Qwen2ForCausalLM(ForCausalLM):
    """Qwen2 family."""
    def __init__(self, config: dict):
        config.setdefault("attention_bias", True)
        config.setdefault("hidden_act", "silu")
        config.setdefault("rope_theta", 1000000.0)
        super().__init__(config)


class Qwen2_5ForCausalLM(ForCausalLM):
    """Qwen2.5 family."""
    def __init__(self, config: dict):
        config.setdefault("attention_bias", True)
        config.setdefault("hidden_act", "silu")
        config.setdefault("rope_theta", 1000000.0)
        super().__init__(config)


class Qwen3ForCausalLM(ForCausalLM):
    """Qwen3 family. Same arch as Qwen2 but with QK-norm."""
    def __init__(self, config: dict):
        config.setdefault("attention_bias", False)
        config.setdefault("hidden_act", "silu")
        config.setdefault("rope_theta", 1000000.0)
        super().__init__(config)
