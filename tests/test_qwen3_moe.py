# tests/test_qwen3_moe.py
import torch
import pytest
from mini_vllm.models.qwen3_moe_model import (
    Qwen3MoeForCausalLM, Qwen3MoeModel, Qwen3MoeSparseMoe,
    Qwen3MoeDecoderLayer, Qwen3MoeExpert,
)


def make_moe_config(**overrides):
    config = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "num_hidden_layers": 2,
        "vocab_size": 100,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 128,
        "rope_theta": 1000000.0,
        "attention_bias": False,
        "hidden_act": "silu",
        # MoE specific
        "num_experts": 4,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 128,
        "norm_topk_prob": True,
        "tie_word_embeddings": False,
    }
    config.update(overrides)
    return config


class TestQwen3MoeExpert:
    def test_expert_forward(self):
        expert = Qwen3MoeExpert(64, 128)
        x = torch.randn(2, 4, 64)
        out = expert(x)
        assert out.shape == (2, 4, 64)

    def test_expert_no_bias(self):
        expert = Qwen3MoeExpert(64, 128)
        assert expert.gate_proj.bias is None
        assert expert.up_proj.bias is None
        assert expert.down_proj.bias is None


class TestQwen3MoeSparseMoe:
    def test_moe_forward(self):
        config = make_moe_config()
        moe = Qwen3MoeSparseMoe(config)
        x = torch.randn(1, 4, 64)
        out = moe(x)
        assert out.shape == (1, 4, 64)

    def test_moe_top1_routing(self):
        config = make_moe_config(num_experts_per_tok=1)
        moe = Qwen3MoeSparseMoe(config)
        x = torch.randn(1, 8, 64)
        out = moe(x)
        assert out.shape == (1, 8, 64)

    def test_moe_top2_routing(self):
        config = make_moe_config(num_experts_per_tok=2)
        moe = Qwen3MoeSparseMoe(config)
        x = torch.randn(1, 8, 64)
        out = moe(x)
        assert out.shape == (1, 8, 64)

    def test_moe_batch(self):
        config = make_moe_config()
        moe = Qwen3MoeSparseMoe(config)
        x = torch.randn(3, 4, 64)
        out = moe(x)
        assert out.shape == (3, 4, 64)

    def test_moe_num_experts(self):
        config = make_moe_config(num_experts=8)
        moe = Qwen3MoeSparseMoe(config)
        assert len(moe.experts) == 8
        x = torch.randn(1, 4, 64)
        out = moe(x)
        assert out.shape == (1, 4, 64)


class TestQwen3MoeDecoderLayer:
    def test_layer_forward(self):
        config = make_moe_config()
        layer = Qwen3MoeDecoderLayer(config, 0)
        x = torch.randn(1, 4, 64)
        cos = torch.ones(1, 1, 4, 16)
        sin = torch.zeros(1, 1, 4, 16)
        out = layer(x, cos, sin)
        assert out.shape == (1, 4, 64)


class TestQwen3MoeForCausalLM:
    def test_forward(self):
        config = make_moe_config()
        model = Qwen3MoeForCausalLM(config)
        input_ids = torch.tensor([[1, 2, 3]])
        positions = torch.tensor([[0, 1, 2]])
        logits = model(input_ids, positions)
        assert logits.shape == (1, 3, 100)

    def test_single_token(self):
        config = make_moe_config()
        model = Qwen3MoeForCausalLM(config)
        logits = model(torch.tensor([[1]]), torch.tensor([[0]]))
        assert logits.shape == (1, 1, 100)

    def test_tied_embeddings(self):
        config = make_moe_config(tie_word_embeddings=True)
        model = Qwen3MoeForCausalLM(config)
        assert model.lm_head.weight is model.model.embed_tokens.weight
        logits = model(torch.tensor([[1, 2]]), torch.tensor([[0, 1]]))
        assert logits.shape == (1, 2, 100)

    def test_untied_embeddings(self):
        config = make_moe_config(tie_word_embeddings=False)
        model = Qwen3MoeForCausalLM(config)
        assert model.lm_head.weight is not model.model.embed_tokens.weight

    def test_matches_real_config(self):
        """Test with config matching the actual minimind-3-moe model."""
        config = make_moe_config(
            hidden_size=768,
            intermediate_size=2432,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=96,
            num_hidden_layers=8,
            vocab_size=6400,
            num_experts=4,
            num_experts_per_tok=1,
            moe_intermediate_size=2432,
            rope_theta=1000000.0,
            max_position_embeddings=32768,
        )
        model = Qwen3MoeForCausalLM(config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        positions = torch.tensor([[0, 1, 2, 3, 4]])
        logits = model(input_ids, positions)
        assert logits.shape == (1, 5, 6400)


class TestLoaderIntegration:
    def test_registered_in_loader(self):
        from mini_vllm.models.loader import _BUILTIN_MODELS
        assert "qwen3_moe" in _BUILTIN_MODELS

    def test_resolve_model_class(self):
        from mini_vllm.models.loader import _resolve_model_class, _ensure_builtin_models, _BUILTIN_MODELS
        _ensure_builtin_models()
        config = make_moe_config()
        model = _resolve_model_class("qwen3_moe", config)
        ExpectedCls = _BUILTIN_MODELS["qwen3_moe"]
        assert isinstance(model, ExpectedCls)
