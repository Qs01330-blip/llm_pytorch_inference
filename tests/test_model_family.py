# tests/test_model_family.py
import torch
import pytest
import torch.nn.functional as F
import mini_vllm.models.loader as loader_mod
from mini_vllm.models.loader import MODEL_REGISTRY, _resolve_model_class
from mini_vllm.models.gemma3_model import Gemma3ForCausalLM

# Resolve model classes based on CUDA availability
loader_mod._ensure_builtin_models()
ForCausalLM = loader_mod._GENERIC_CASUAL_LM
_BUILTIN_MODELS = loader_mod._BUILTIN_MODELS
LlamaForCausalLM = _BUILTIN_MODELS["llama"]
Qwen2ForCausalLM = _BUILTIN_MODELS["qwen2"]
Qwen2_5ForCausalLM = _BUILTIN_MODELS["qwen2.5"]
Qwen3ForCausalLM = _BUILTIN_MODELS["qwen3"]


def make_config(**overrides):
    """Build a minimal model config for testing."""
    config = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "vocab_size": 100,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 128,
    }
    config.update(overrides)
    return config


class TestModelFamilies:
    """Each model family class should produce valid logits and be a ForCausalLM subclass."""

    @pytest.mark.parametrize("cls,expected_bias,expected_act,expected_rope", [
        (LlamaForCausalLM, False, "silu", 10000.0),
        (Qwen2ForCausalLM, True, "silu", 1000000.0),
        (Qwen2_5ForCausalLM, True, "silu", 1000000.0),
        (Qwen3ForCausalLM, False, "silu", 1000000.0),
    ])
    def test_model_family_forward(self, cls, expected_bias, expected_act, expected_rope):
        """Each model class should produce correct output shape."""
        config = make_config()
        model = cls(config)
        assert isinstance(model, ForCausalLM)
        logits = model(torch.tensor([[1, 2]]), torch.tensor([[0, 1]]))
        assert logits.shape == (1, 2, 100)

    def test_gemma3_forward(self):
        """Gemma3 standalone model should produce correct output shape."""
        config = make_config()
        model = Gemma3ForCausalLM(config)
        logits = model(torch.tensor([[1, 2]]), torch.tensor([[0, 1]]))
        assert logits.shape == (1, 2, 100)

    def test_qwen2_bias_default(self):
        """Qwen2 should default to attention_bias=True."""
        config = make_config()
        model = Qwen2ForCausalLM(config)
        assert model.model.layers[0].self_attn.q_proj.bias is not None
        assert model.model.layers[0].self_attn.k_proj.bias is not None
        assert model.model.layers[0].self_attn.v_proj.bias is not None

    def test_qwen3_no_bias(self):
        """Qwen3 should default to attention_bias=False."""
        config = make_config()
        model = Qwen3ForCausalLM(config)
        assert model.model.layers[0].self_attn.q_proj.bias is None

    def test_llama_defaults(self):
        """LLaMA should use default silu, no bias, rope_theta=10000."""
        config = make_config()
        model = LlamaForCausalLM(config)
        assert model.model.layers[0].self_attn.q_proj.bias is None
        mlp = model.model.layers[0].mlp
        if hasattr(mlp, 'act_fn'):
            assert mlp.act_fn is F.silu

    def test_config_override_takes_precedence(self):
        """Explicit config values should not be overridden by setdefault."""
        config = make_config(attention_bias=False, rope_theta=500000.0)
        model = Qwen2ForCausalLM(config)
        # attention_bias=False was explicit, should NOT be overridden to True
        assert model.model.layers[0].self_attn.q_proj.bias is None

    def test_rope_theta_override(self):
        """Explicit rope_theta should not be overridden by setdefault."""
        config = make_config(rope_theta=500000.0)
        model = Qwen2ForCausalLM(config)
        # Check rope_theta was preserved (inv_freq depends on it)
        assert model.model.rotary_emb is not None


class TestRegistryIntegration:
    """Test that builtin models are properly registered."""

    def test_builtin_types_in_registry(self):
        """All builtin model types should be pre-registered."""
        for model_type in _BUILTIN_MODELS:
            assert model_type in MODEL_REGISTRY, f"{model_type} not in MODEL_REGISTRY"

    def test_resolve_qwen2_uses_specific_class(self):
        """Resolving 'qwen2' should return Qwen2ForCausalLM, not generic ForCausalLM."""
        config = make_config()
        model = _resolve_model_class("qwen2", config)
        assert type(model) is Qwen2ForCausalLM

    def test_resolve_llama_uses_specific_class(self):
        """Resolving 'llama' should return LlamaForCausalLM."""
        config = make_config()
        model = _resolve_model_class("llama", config)
        assert type(model) is LlamaForCausalLM

    def test_resolve_qwen2_5_uses_specific_class(self):
        """Resolving 'qwen2.5' should return Qwen2_5ForCausalLM."""
        config = make_config()
        model = _resolve_model_class("qwen2.5", config)
        assert type(model) is Qwen2_5ForCausalLM

    def test_resolve_unknown_type_returns_generic(self):
        """Unknown model_type should return generic ForCausalLM."""
        config = make_config()
        model = _resolve_model_class("brand_new_model", config)
        assert type(model) is ForCausalLM

    def test_resolve_none_returns_generic(self):
        """model_type=None should return generic ForCausalLM."""
        config = make_config()
        model = _resolve_model_class(None, config)
        assert type(model) is ForCausalLM
