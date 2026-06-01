# tests/test_model.py
import torch
import pytest
import mini_vllm.models.loader as loader_mod
from mini_vllm.models.loader import _detect_config_from_weights, register_model, MODEL_REGISTRY, _resolve_model_class
from mini_vllm.models.model import MLP, Attention

# Resolve the correct ForCausalLM based on CUDA availability
loader_mod._ensure_builtin_models()
ForCausalLM = loader_mod._GENERIC_CASUAL_LM


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
        "rope_theta": 10000.0,
    }
    config.update(overrides)
    return config


def test_generic_model_default():
    """Default config (LLaMA-like: no bias, silu)"""
    config = make_config()
    model = ForCausalLM(config)
    input_ids = torch.tensor([[1, 2, 3]])
    positions = torch.tensor([[0, 1, 2]])
    logits = model(input_ids, positions)
    assert logits.shape == (1, 3, 100)


def test_generic_model_qwen2():
    """Qwen2 config: attention_bias=True, rope_theta=1000000"""
    config = make_config(attention_bias=True, rope_theta=1000000.0)
    model = ForCausalLM(config)
    input_ids = torch.tensor([[1, 2]])
    positions = torch.tensor([[0, 1]])
    logits = model(input_ids, positions)
    assert logits.shape == (1, 2, 100)



def test_mlp_activations():
    """Test different activation functions."""
    for act in ["silu", "gelu", "relu"]:
        mlp = MLP(64, 128, hidden_act=act)
        x = torch.randn(1, 4, 64)
        out = mlp(x)
        assert out.shape == (1, 4, 64)


def test_mlp_unsupported_activation():
    """Unsupported activation should raise."""
    with pytest.raises(ValueError, match="Unsupported activation"):
        MLP(64, 128, hidden_act="swish")


def test_attention_bias_true():
    """Attention with bias (Qwen2-style)."""
    attn = Attention(64, 4, 2, 16, attention_bias=True)
    x = torch.randn(1, 4, 64)
    cos = torch.ones(1, 1, 4, 16)
    sin = torch.zeros(1, 1, 4, 16)
    out = attn(x, cos, sin)
    assert out.shape == (1, 4, 64)
    # Verify bias exists on q/k/v
    assert attn.q_proj.bias is not None
    assert attn.k_proj.bias is not None
    assert attn.v_proj.bias is not None
    assert attn.o_proj.bias is None


def test_attention_bias_false():
    """Attention without bias (LLaMA-style)."""
    attn = Attention(64, 4, 2, 16, attention_bias=False)
    x = torch.randn(1, 4, 64)
    cos = torch.ones(1, 1, 4, 16)
    sin = torch.zeros(1, 1, 4, 16)
    out = attn(x, cos, sin)
    assert out.shape == (1, 4, 64)
    assert attn.q_proj.bias is None
    assert attn.k_proj.bias is None
    assert attn.v_proj.bias is None
    assert attn.o_proj.bias is None


def test_model_mha():
    """Multi-head attention (num_kv_heads == num_heads)."""
    config = make_config(num_key_value_heads=4)
    model = ForCausalLM(config)
    logits = model(torch.tensor([[1]]), torch.tensor([[0]]))
    assert logits.shape == (1, 1, 100)


def test_model_mqa():
    """Multi-query attention (num_kv_heads == 1)."""
    config = make_config(num_key_value_heads=1)
    model = ForCausalLM(config)
    logits = model(torch.tensor([[1, 2]]), torch.tensor([[0, 1]]))
    assert logits.shape == (1, 2, 100)


def test_config_defaults_applied():
    """Config with missing optional keys should still work."""
    config = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "vocab_size": 50,
    }
    # These should be set by _apply_config_defaults or config.get defaults
    model = ForCausalLM(config)
    logits = model(torch.tensor([[1]]), torch.tensor([[0]]))
    assert logits.shape == (1, 1, 50)


class TestWeightAutoDetection:
    """Test auto-detection of model architecture from checkpoint weights."""

    def test_detect_bias_from_weights(self):
        """Qwen2-style: config says no bias, but weights have bias."""
        state_dict = {
            "model.layers.0.self_attn.q_proj.bias": torch.zeros(64),
            "model.layers.0.self_attn.k_proj.bias": torch.zeros(32),
            "model.layers.0.self_attn.v_proj.bias": torch.zeros(32),
            "model.embed_tokens.weight": torch.zeros(100, 64),
        }
        config = {"attention_bias": False}  # Config says False
        _detect_config_from_weights(state_dict, config)
        assert config["attention_bias"] is True  # Should be overridden

    def test_detect_no_bias_from_weights(self):
        """LLaMA-style: config says no bias, weights confirm."""
        state_dict = {
            "model.embed_tokens.weight": torch.zeros(100, 64),
        }
        config = {"attention_bias": False}
        _detect_config_from_weights(state_dict, config)
        assert config["attention_bias"] is False

    def test_detect_o_proj_bias(self):
        """Some models have bias on o_proj."""
        state_dict = {
            "model.layers.0.self_attn.o_proj.bias": torch.zeros(64),
        }
        config = {}
        _detect_config_from_weights(state_dict, config)
        assert config.get("_o_proj_bias") is True

    def test_model_with_detected_bias(self):
        """Build model with auto-detected bias and verify forward works."""
        config = make_config(attention_bias=True)
        model = ForCausalLM(config)
        logits = model(torch.tensor([[1, 2]]), torch.tensor([[0, 1]]))
        assert logits.shape == (1, 2, 100)
        # Verify bias parameters exist
        layer0_attn = model.model.layers[0].self_attn
        assert layer0_attn.q_proj.bias is not None
        assert layer0_attn.o_proj.bias is None  # o_proj default: no bias


def test_generic_model_qwen3():
    """Qwen3 config: attention_bias=False, rope_theta=1000000 (same arch as Qwen2, different config)."""
    config = make_config(attention_bias=False, rope_theta=1000000.0)
    model = ForCausalLM(config)
    input_ids = torch.tensor([[1, 2, 3]])
    positions = torch.tensor([[0, 1, 2]])
    logits = model(input_ids, positions)
    assert logits.shape == (1, 3, 100)
    # Qwen3: no bias on q/k/v
    layer0_attn = model.model.layers[0].self_attn
    assert layer0_attn.q_proj.bias is None
    assert layer0_attn.k_proj.bias is None
    assert layer0_attn.v_proj.bias is None


def test_model_gqa():
    """Grouped-query attention (num_kv_heads < num_heads, e.g. LLaMA-3 8B: 32 heads, 8 kv_heads)."""
    config = make_config(num_attention_heads=4, num_key_value_heads=2)
    model = ForCausalLM(config)
    logits = model(torch.tensor([[1, 2]]), torch.tensor([[0, 1]]))
    assert logits.shape == (1, 2, 100)
    # Verify GQA structure
    layer0_attn = model.model.layers[0].self_attn
    assert layer0_attn.num_kv_groups == 2  # 4 heads / 2 kv_heads


def test_attention_o_proj_bias():
    """Attention with o_proj bias (some models like Phi-3)."""
    attn = Attention(64, 4, 2, 16, attention_bias=True, o_proj_bias=True)
    x = torch.randn(1, 4, 64)
    cos = torch.ones(1, 1, 4, 16)
    sin = torch.zeros(1, 1, 4, 16)
    out = attn(x, cos, sin)
    assert out.shape == (1, 4, 64)
    assert attn.q_proj.bias is not None
    assert attn.o_proj.bias is not None


def test_tied_embeddings():
    """Model with tied embeddings (lm_head shares embed_tokens weights)."""
    config = make_config()
    model = ForCausalLM(config)
    # Simulate tied embeddings: copy embed weights to lm_head
    model.lm_head.weight = model.model.embed_tokens.weight
    logits = model(torch.tensor([[1, 2]]), torch.tensor([[0, 1]]))
    assert logits.shape == (1, 2, 100)
    # Verify weights are actually shared
    assert model.lm_head.weight is model.model.embed_tokens.weight



class TestRegisterModel:
    """Test custom model registration."""

    def setup_method(self):
        """Clean up registry before each test."""
        self._original = MODEL_REGISTRY.copy()

    def teardown_method(self):
        """Restore registry after each test."""
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(self._original)

    def test_register_custom_type(self):
        """Register a custom model type with config overrides."""
        register_model("my_model", config_overrides={"attention_bias": True, "rope_theta": 500000.0})
        assert "my_model" in MODEL_REGISTRY
        cls, overrides = MODEL_REGISTRY["my_model"]
        assert cls is ForCausalLM  # default class
        assert overrides["attention_bias"] is True
        assert overrides["rope_theta"] == 500000.0

    def test_resolve_registered_model(self):
        """Registered model type should use ForCausalLM with overrides."""
        register_model("custom_llm", config_overrides={"attention_bias": True})
        config = make_config()
        model = _resolve_model_class("custom_llm", config)
        assert isinstance(model, ForCausalLM)
        assert config["attention_bias"] is True

    def test_resolve_known_type(self):
        """Known model type should resolve without registration."""
        config = make_config()
        model = _resolve_model_class("llama", config)
        assert isinstance(model, ForCausalLM)

    def test_resolve_unknown_type_with_warning(self):
        """Unknown model type should still work with a warning."""
        config = make_config()
        model = _resolve_model_class("brand_new_model", config)
        assert isinstance(model, ForCausalLM)
