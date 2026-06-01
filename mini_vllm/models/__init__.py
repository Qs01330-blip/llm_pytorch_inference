from mini_vllm.models.model import (
    RMSNorm,
    RotaryEmbedding,
    MLP,
    Attention,
    DecoderLayer,
    TransformerModel,
    ForCausalLM,
)
from mini_vllm.models.gemma3_model import Gemma3ForCausalLM
from mini_vllm.models.qwen3_5_model import Qwen3_5ForCausalLM
from mini_vllm.models.qwen3_moe_model import Qwen3MoeForCausalLM
from mini_vllm.models.loader import load_model, register_model, SUPPORTED_MODEL_TYPES


def __getattr__(name):
    """Lazy-resolve CUDA/PyTorch model classes on first access."""
    from mini_vllm.models.loader import _ensure_builtin_models, _BUILTIN_MODELS
    _dynamic = {
        "LlamaForCausalLM": "llama",
        "Qwen2ForCausalLM": "qwen2",
        "Qwen2_5ForCausalLM": "qwen2.5",
        "Qwen3ForCausalLM": "qwen3",
    }
    if name in _dynamic:
        _ensure_builtin_models()
        return _BUILTIN_MODELS[_dynamic[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RMSNorm", "RotaryEmbedding", "MLP", "Attention",
    "DecoderLayer", "TransformerModel", "ForCausalLM",
    "LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen2_5ForCausalLM", "Qwen3ForCausalLM",
    "Gemma3ForCausalLM", "Qwen3_5ForCausalLM", "Qwen3MoeForCausalLM",
    "load_model", "register_model", "SUPPORTED_MODEL_TYPES",
]
