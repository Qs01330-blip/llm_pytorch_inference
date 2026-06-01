# mini_vllm/models/loader.py
import json
import torch
from pathlib import Path
from safetensors.torch import load_file
from mini_vllm.utils.logger import logger


def _make_standard_model(base_cls):
    """Create standard model subclasses dynamically from a base ForCausalLM class."""
    class LlamaForCausalLM(base_cls):
        def __init__(self, config: dict):
            config.setdefault("attention_bias", False)
            config.setdefault("hidden_act", "silu")
            config.setdefault("rope_theta", 10000.0)
            super().__init__(config)

    class Qwen2ForCausalLM(base_cls):
        def __init__(self, config: dict):
            config.setdefault("attention_bias", True)
            config.setdefault("hidden_act", "silu")
            config.setdefault("rope_theta", 1000000.0)
            super().__init__(config)

    class Qwen2_5ForCausalLM(base_cls):
        def __init__(self, config: dict):
            config.setdefault("attention_bias", True)
            config.setdefault("hidden_act", "silu")
            config.setdefault("rope_theta", 1000000.0)
            super().__init__(config)

    class Qwen3ForCausalLM(base_cls):
        def __init__(self, config: dict):
            config.setdefault("attention_bias", False)
            config.setdefault("hidden_act", "silu")
            config.setdefault("rope_theta", 1000000.0)
            super().__init__(config)

    return LlamaForCausalLM, Qwen2ForCausalLM, Qwen2_5ForCausalLM, Qwen3ForCausalLM


def _select_model_classes():
    """Select CUDA or PyTorch model classes based on CUDA extension availability."""
    from mini_vllm.ops._utils import is_ext_loaded

    if is_ext_loaded():
        from mini_vllm.models.model_cuda import ForCausalLM
    else:
        from mini_vllm.models.model import ForCausalLM

    # Create standard model subclasses from the selected base class
    LlamaForCausalLM, Qwen2ForCausalLM, Qwen2_5ForCausalLM, Qwen3ForCausalLM = _make_standard_model(ForCausalLM)

    # Specialized models: use CUDA versions when extension is loaded
    # Qwen3.5 always uses PyTorch version (no CUDA variant)
    from mini_vllm.models.qwen3_5_model import Qwen3_5ForCausalLM
    if is_ext_loaded():
        from mini_vllm.models.gemma3_cuda import Gemma3ForCausalLM
        from mini_vllm.models.qwen3_moe_cuda import Qwen3MoeForCausalLM
    else:
        from mini_vllm.models.gemma3_model import Gemma3ForCausalLM
        from mini_vllm.models.qwen3_moe_model import Qwen3MoeForCausalLM

    return {
        "llama": LlamaForCausalLM,
        "qwen2": Qwen2ForCausalLM,
        "qwen2.5": Qwen2_5ForCausalLM,
        "qwen3": Qwen3ForCausalLM,
        "qwen3_moe": Qwen3MoeForCausalLM,
        "gemma3_text": Gemma3ForCausalLM,
        "qwen3_5_text": Qwen3_5ForCausalLM,
        "internlm": ForCausalLM,
        "internlm2": ForCausalLM,
        "yi": ForCausalLM,
        "deepseek": ForCausalLM,
        "deepseek_v2": ForCausalLM,
        "cohere": ForCausalLM,
        "command-r": ForCausalLM,
    }, ForCausalLM


# Model registry: maps model_type -> (class, config_overrides)
MODEL_REGISTRY: dict[str, tuple[type, dict]] = {}

# Lazy-init builtin models (resolved on first load)
_BUILTIN_MODELS: dict[str, type] = {}
_GENERIC_CASUAL_LM: type | None = None


def _ensure_builtin_models():
    global _GENERIC_CASUAL_LM, SUPPORTED_MODEL_TYPES
    if not _BUILTIN_MODELS:
        models, _GENERIC_CASUAL_LM = _select_model_classes()
        _BUILTIN_MODELS.update(models)
        # Pre-populate registry
        for _type, _cls in _BUILTIN_MODELS.items():
            register_model(_type, cls=_cls)
        SUPPORTED_MODEL_TYPES.update(_BUILTIN_MODELS.keys())

# Kept for backward compatibility
SUPPORTED_MODEL_TYPES: set[str] = set()


def register_model(model_type: str, cls: type = None, config_overrides: dict = None):
    """Register a custom model type with optional class and config overrides.

    Args:
        model_type: The model_type string from config.json
        cls: Custom nn.Module class (defaults to generic ForCausalLM)
        config_overrides: Dict of config values to override/patch
    """
    _ensure_builtin_models()
    MODEL_REGISTRY[model_type] = (cls or _GENERIC_CASUAL_LM, config_overrides or {})


def _detect_config_from_weights(state_dict: dict, model_config: dict):
    """Auto-detect model architecture from checkpoint weights.

    Some models (e.g. Qwen2/Qwen2.5) have attention_bias=True in weights
    but attention_bias=false in config.json. We detect from the actual weights.
    """
    # Detect attention bias from weight keys
    # If "model.layers.0.self_attn.q_proj.bias" exists in weights, the model uses bias
    q_bias_key = "model.layers.0.self_attn.q_proj.bias"
    if q_bias_key in state_dict:
        if not model_config.get("attention_bias", False):
            logger.info("Auto-detected attention_bias=True from weights (overriding config)")
            model_config["attention_bias"] = True
    else:
        model_config.setdefault("attention_bias", False)

    # Detect o_proj bias
    o_bias_key = "model.layers.0.self_attn.o_proj.bias"
    if o_bias_key in state_dict:
        model_config["_o_proj_bias"] = True

    # Detect QK-norm (Qwen3, Gemma 3, etc.)
    q_norm_key = "model.layers.0.self_attn.q_norm.weight"
    if q_norm_key in state_dict:
        model_config["_qk_norm"] = True
        logger.info("Auto-detected qk_norm=True from weights")

    # Detect feedforward layernorms (Gemma 3)
    pre_ff_key = "model.layers.0.pre_feedforward_layernorm.weight"
    if pre_ff_key in state_dict:
        model_config["_has_feedforward_layernorms"] = True
        logger.info("Auto-detected feedforward_layernorms=True from weights")

    # Map hidden_activation -> hidden_act (Gemma 3 uses hidden_activation)
    if "hidden_activation" in model_config and "hidden_act" not in model_config:
        _ACT_MAP = {
            "gelu_pytorch_tanh": "gelu_new",
            "gelu": "gelu",
            "silu": "silu",
        }
        ha = model_config["hidden_activation"]
        model_config["hidden_act"] = _ACT_MAP.get(ha, ha)
        logger.info(f"Mapped hidden_activation={ha} -> hidden_act={model_config['hidden_act']}")


def _resolve_model_class(model_type: str, model_config: dict):
    """Resolve model class and apply config defaults."""
    _ensure_builtin_models()
    # 1. Check registry (covers both builtin and user-registered)
    if model_type in MODEL_REGISTRY:
        cls, overrides = MODEL_REGISTRY[model_type]
        model_config.update(overrides)
        return cls(model_config)

    # 2. model_type=None: use generic ForCausalLM
    if model_type is None:
        _apply_config_defaults(model_config)
        return _GENERIC_CASUAL_LM(model_config)

    # 3. Unknown type: try generic with warning
    logger.warning(
        f"Unknown model_type '{model_type}'. Attempting with generic implementation. "
        f"Known types: {sorted(_BUILTIN_MODELS.keys())}. "
        f"Use register_model() to add custom types."
    )
    _apply_config_defaults(model_config)
    return _GENERIC_CASUAL_LM(model_config)


def _apply_config_defaults(model_config: dict):
    """Set sensible defaults for config keys that may be missing."""
    model_config.setdefault("attention_bias", False)
    model_config.setdefault("hidden_act", "silu")
    model_config.setdefault("rms_norm_eps", 1e-6)


def _load_weights(path: Path) -> dict:
    """Load model weights from safetensors or PyTorch format."""
    safetensors_path = path / "model.safetensors"
    if safetensors_path.exists():
        return load_file(str(safetensors_path))

    # Try sharded loading
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        state_dict = {}
        for shard_file in set(index["weight_map"].values()):
            shard_path = path / shard_file
            state_dict.update(load_file(str(shard_path)))
        return state_dict

    # Fallback to PyTorch format
    bin_files = list(path.glob("*.bin"))
    if bin_files:
        state_dict = {}
        for bf in sorted(bin_files):
            state_dict.update(torch.load(str(bf), map_location="cpu", weights_only=True))
        return state_dict

    raise FileNotFoundError(f"No model weights found in {path}")


def load_model(model_path: str, device: str = "auto", dtype: str = "float16") -> tuple:
    """Load a HuggingFace format model with auto-detection.

    Loading order: config.json → weights → auto-detect architecture → build model → load weights.
    This ensures the model architecture matches the actual checkpoint weights.
    """
    # Resolve "auto" device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    path = Path(model_path)

    # 1. Read config.json
    config_path = path / "config.json"
    with open(config_path) as f:
        model_config = json.load(f)

    # Handle nested config (e.g. Qwen3.5 has text_config + vision_config)
    model_type = model_config.get("model_type", None)
    if model_type == "qwen3_5" and "text_config" in model_config:
        text_config = model_config["text_config"]
        # Flatten text_config to top level for our model implementation
        model_config = {**model_config, **text_config}
        model_type = "qwen3_5_text"
        model_config["model_type"] = model_type
        # Extract rope_parameters into flat config keys
        rope_params = model_config.get("rope_parameters", {})
        model_config["_rope_theta"] = rope_params.get("rope_theta", 10000000.0)
        model_config["_partial_rotary_factor"] = rope_params.get("partial_rotary_factor", 0.25)
        model_config["_mrope_interleaved"] = rope_params.get("mrope_interleaved", False)
        # Store layer_types for hybrid attention routing
        model_config["_layer_types"] = model_config.get("layer_types", ["full_attention"] * model_config.get("num_hidden_layers", 1))
        logger.info(f"Qwen3.5 nested config: flattened text_config, {sum(1 for t in model_config['_layer_types'] if t == 'linear_attention')} linear + {sum(1 for t in model_config['_layer_types'] if t == 'full_attention')} full attention layers")

    # 2. Load weights FIRST to detect architecture
    state_dict = _load_weights(path)
    mapped_state_dict = _map_weights(state_dict, model_config)

    # 2.5 Fuse gate_proj + up_proj weights for FusedMLP
    # Qwen3.5 has no CUDA variant, its MLP uses separate gate/up proj
    from mini_vllm.ops._utils import is_ext_loaded
    if is_ext_loaded() and model_type != "qwen3_5_text":
        from mini_vllm.ops.mlp import fuse_mlp_weights
        mapped_state_dict = fuse_mlp_weights(mapped_state_dict)

    # 3. Auto-detect architecture from weights
    _detect_config_from_weights(mapped_state_dict, model_config)

    # Warn about unsupported features
    if "rope_scaling" in model_config:
        logger.warning(f"rope_scaling detected but not supported yet (may affect very long contexts)")

    # 4. Handle tied embeddings
    if "lm_head.weight" not in mapped_state_dict:
        embed_key = "model.embed_tokens.weight"
        if embed_key in mapped_state_dict:
            mapped_state_dict["lm_head.weight"] = mapped_state_dict[embed_key]
            logger.info("Tied embeddings: lm_head.weight copied from embed_tokens.weight")
        else:
            logger.warning("WARNING: lm_head.weight not found and embed_tokens.weight missing!")

    # 5. Resolve model class (now with correct config)
    model = _resolve_model_class(model_type, model_config)

    # 6. Load weights into model
    missing, unexpected = model.load_state_dict(mapped_state_dict, strict=False)
    if missing:
        if model_config.get("tie_word_embeddings", False):
            missing = [k for k in missing if k != "lm_head.weight"]
        if missing:
            logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.debug(f"Unexpected keys: {unexpected}")

    # 7. Move to device
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.float16)
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()

    # Print startup banner — check if THIS model actually uses CUDA ops
    mro_modules = [c.__module__ for c in type(model).__mro__]
    uses_cuda = any("cuda" in m for m in mro_modules)
    op_backend = "CUDA ops" if uses_cuda else "PyTorch"
    device_name = "GPU (CUDA)" if device == "cuda" else "CPU"

    num_layers = model_config.get("num_hidden_layers", "?")
    num_heads = model_config.get("num_attention_heads", "?")
    num_kv = model_config.get("num_key_value_heads", num_heads)
    hidden = model_config.get("hidden_size", "?")
    head_dim = model_config.get("head_dim", hidden // num_heads if isinstance(hidden, int) and isinstance(num_heads, int) else "?")
    rope_theta = model_config.get("_rope_theta", model_config.get("rope_theta", 10000.0))
    attn_bias = "Yes" if model_config.get("attention_bias", False) else "No"
    qk_norm = "Yes" if model_config.get("_qk_norm", False) else "No"
    tied = "Yes" if model_config.get("tie_word_embeddings", False) else "No"

    w = 56
    top = "+" + "=" * (w - 2) + "+"
    mid = "+" + "-" * (w - 2) + "+"
    bot = "+" + "=" * (w - 2) + "+"

    def row(label, value):
        content = f"  {label:<16}: {value}"
        # pad content to w-2 chars, then wrap with |
        logger.info(f"|{content:<{w-2}}|")

    logger.info("")
    logger.info(top)
    logger.info(f"|{'mini-vllm Inference Engine':^{w-2}}|")
    logger.info(mid)
    row("Model Path", model_path)
    row("Backend", op_backend)
    row("Device", device_name)
    row("Model Class", type(model).__name__)
    logger.info(mid)
    row("Model Type", str(model_type or 'unknown'))
    row("Layers", str(num_layers))
    row("Hidden Size", str(hidden))
    row("Attn Heads", str(num_heads))
    row("KV Heads", str(num_kv))
    row("Head Dim", str(head_dim))
    row("RoPE Theta", str(rope_theta))
    row("Attn Bias", attn_bias)
    row("QK Norm", qk_norm)
    row("Tied Embedding", tied)
    row("Dtype", str(dtype))
    logger.info(bot)
    logger.info("")

    return model, model_config


def _map_weights(state_dict: dict, config: dict) -> dict:
    """Map HuggingFace transformers weight names to our model structure.

    Works for any model using the standard 'model.layers.X.*' naming convention
    (Qwen2, Qwen3, LLaMA, Mistral, Phi, etc.).
    """
    mapped = {}
    model_type = config.get("model_type", "")

    for key, value in state_dict.items():
        # Skip vision model weights (text-only inference)
        if "visual." in key:
            continue
        # Skip MTP (multi-token prediction) weights
        if key.startswith("mtp."):
            continue

        # Qwen3.5: model.language_model.* → model.*
        if model_type == "qwen3_5_text" and key.startswith("model.language_model."):
            new_key = "model." + key[len("model.language_model."):]
            mapped[new_key] = value
        elif key == "lm_head.weight":
            mapped[key] = value
        elif key.startswith("model."):
            mapped[key] = value
        else:
            mapped[f"model.{key}"] = value

    return mapped
