# mini_vllm/models/model.py
# Generic transformer model implementation - adapts to any HuggingFace config.json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, plus_one: bool = False):
        super().__init__()
        # plus_one=True: use (1 + weight) formula with weight init to zeros (Gemma3)
        # plus_one=False: use weight formula with weight init to ones (standard)
        self.weight = Parameter(torch.zeros(hidden_size) if plus_one else torch.ones(hidden_size))
        self.eps = eps
        self.plus_one = plus_one

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.plus_one:
            return (x * (1.0 + self.weight.float())).to(orig_dtype)
        return (self.weight * x).to(orig_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0, rope_scaling: dict | None = None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        # Apply Llama 3 rope_scaling if present
        rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type") if rope_scaling else None
        if rope_type == "llama3":
            factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            old_context_len = rope_scaling["original_max_position_embeddings"]

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            wavelen = 2 * math.pi / inv_freq
            # Low frequency: divide by factor
            inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
            # Medium frequency: smooth interpolation
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
            inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim
        # RoPE cache: avoid recomputing cos/sin every forward pass
        self._cos_cache: torch.Tensor | None = None
        self._sin_cache: torch.Tensor | None = None
        self._cache_len: int = 0

    def forward(self, seq_len_or_positions, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Accept either int or tensor for seq_len
        if isinstance(seq_len_or_positions, torch.Tensor):
            seq_len = seq_len_or_positions.max().item() + 1
            if device is None:
                device = seq_len_or_positions.device
        else:
            seq_len = seq_len_or_positions
            if device is None:
                device = self.inv_freq.device
        # Return cached cos/sin if large enough
        if self._cos_cache is not None and self._cache_len >= seq_len and self._cos_cache.device == device:
            return self._cos_cache[:seq_len], self._sin_cache[:seq_len]
        # Compute and cache (grow to seq_len, rounded up to reduce reallocs)
        grow_to = max(seq_len, self._cache_len * 2, 256)
        t = torch.arange(grow_to, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cache = emb.cos()
        self._sin_cache = emb.sin()
        self._cache_len = grow_to
        return self._cos_cache[:seq_len], self._sin_cache[:seq_len]



_ACT_FN_REGISTRY = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
}



def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        if hidden_act not in _ACT_FN_REGISTRY:
            raise ValueError(f"Unsupported activation: {hidden_act}. Supported: {list(_ACT_FN_REGISTRY.keys())}")
        self.act_fn = _ACT_FN_REGISTRY[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        attention_bias: bool = False,
        o_proj_bias: bool = False,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        query_pre_attn_scalar: float | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self._scale = query_pre_attn_scalar if query_pre_attn_scalar is not None else head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=o_proj_bias)

        # QK-norm: per-head RMSNorm on Q and K
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        block_tables: list[int] | list[list[int]] | None = None,
        slot_idx: int | list[int] | None = None,
        layer_idx: int = 0,
        use_cache: bool = False,
        num_cached_tokens: int = 0,
        num_cached_tokens_list: list[int] | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK-norm: per-head RMSNorm (Qwen3, etc.)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Apply RoPE to both Q and K (K needs RoPE before writing to cache)
        q, k = apply_rotary_pos_emb(q, k, position_cos, position_sin)

        # Write to KV Cache
        if kv_cache is not None and block_tables is not None and slot_idx is not None:
            self._write_kv_cache(kv_cache, k, v, block_tables, slot_idx, layer_idx)
            # Read all historical K, V from cache (only during decode, not prefill)
            # Cache already contains RoPE-applied K from all previous writes
            if use_cache:
                if num_cached_tokens_list is not None:
                    # Batch decode: per-sequence read from cache
                    k, v = self._read_kv_cache_batch(kv_cache, block_tables, layer_idx, num_cached_tokens_list, hidden_states.device)
                else:
                    # Single sequence: read all blocks
                    k, v = self._read_kv_cache(kv_cache, block_tables, layer_idx)
                    # +1 to include the token just written above
                    k = k[:num_cached_tokens + 1]
                    v = v[:num_cached_tokens + 1]
                    k = k.unsqueeze(0).expand(batch_size, -1, -1, -1).transpose(1, 2)
                    v = v.unsqueeze(0).expand(batch_size, -1, -1, -1).transpose(1, 2)

        # GQA: expand KV heads
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        scale = math.sqrt(self._scale)
        if attn_mask is not None:
            # Batch decode: explicit mask handles variable cache lengths per sequence
            attn_output = F.scaled_dot_product_attention(q, k, v, scale=1.0 / scale, attn_mask=attn_mask)
        elif use_cache:
            # Single-sequence decode: S_q=1, attend to all cached positions
            # Cannot use is_causal=True because PyTorch SDPA with S_q=1 only
            # allows attending to position 0 (it builds a S_k x S_k causal mask)
            attn_output = F.scaled_dot_product_attention(q, k, v, scale=1.0 / scale)
        else:
            # Prefill: causal mask (all sequences have same length)
            attn_output = F.scaled_dot_product_attention(q, k, v, scale=1.0 / scale, is_causal=True)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

    def _write_kv_cache(self, kv_cache, k, v, block_tables, slot_idx, layer_idx):
        block_size = kv_cache.shape[3]
        # Support both single (prefill) and batch (decode) block_tables
        is_batch = isinstance(block_tables[0], list)
        for i in range(k.shape[2]):
            if is_batch:
                for b in range(k.shape[0]):
                    pos = slot_idx[b] + i
                    block_idx = block_tables[b][pos // block_size]
                    block_offset = pos % block_size
                    kv_cache[layer_idx, 0, block_idx, block_offset] = k[b, :, i]
                    kv_cache[layer_idx, 1, block_idx, block_offset] = v[b, :, i]
            else:
                pos = slot_idx + i
                block_idx = block_tables[pos // block_size]
                block_offset = pos % block_size
                kv_cache[layer_idx, 0, block_idx, block_offset] = k[0, :, i]
                kv_cache[layer_idx, 1, block_idx, block_offset] = v[0, :, i]

    def _read_kv_cache(self, kv_cache, block_tables, layer_idx):
        k_list = []
        v_list = []
        for block_idx in block_tables:
            k_list.append(kv_cache[layer_idx, 0, block_idx])
            v_list.append(kv_cache[layer_idx, 1, block_idx])
        k = torch.cat(k_list, dim=0)
        v = torch.cat(v_list, dim=0)
        return k, v

    def _read_kv_cache_batch(self, kv_cache, block_tables_list, layer_idx, num_cached_tokens_list, device):
        """Read KV cache for batch decode, pad to same length for batching."""
        batch_size = len(block_tables_list)
        max_cached = max(nc + 1 for nc in num_cached_tokens_list)

        k_list = []
        v_list = []
        for i in range(batch_size):
            k_seq, v_seq = self._read_kv_cache(kv_cache, block_tables_list[i], layer_idx)
            # Truncate to actual cached tokens + 1 (include just-written token)
            k_seq = k_seq[:num_cached_tokens_list[i] + 1]
            v_seq = v_seq[:num_cached_tokens_list[i] + 1]
            # Pad to max_cached for uniform batch shape
            if k_seq.shape[0] < max_cached:
                pad_size = max_cached - k_seq.shape[0]
                k_seq = torch.cat([k_seq, torch.zeros(pad_size, *k_seq.shape[1:], device=device, dtype=k_seq.dtype)], dim=0)
                v_seq = torch.cat([v_seq, torch.zeros(pad_size, *v_seq.shape[1:], device=device, dtype=v_seq.dtype)], dim=0)
            k_list.append(k_seq)
            v_list.append(v_seq)

        k = torch.stack(k_list, dim=0)  # [batch, max_cached, num_kv_heads, head_dim]
        v = torch.stack(v_list, dim=0)
        k = k.transpose(1, 2)  # [batch, num_kv_heads, max_cached, head_dim]
        v = v.transpose(1, 2)
        return k, v


class DecoderLayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden_size = config["hidden_size"]
        attention_bias = config.get("attention_bias", False)
        hidden_act = config.get("hidden_act", "silu")
        rms_plus_one = config.get("_rms_norm_plus_one", False)

        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_heads=config["num_attention_heads"],
            num_kv_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
            head_dim=config.get("head_dim", hidden_size // config["num_attention_heads"]),
            attention_bias=attention_bias,
            o_proj_bias=config.get("_o_proj_bias", False),
            qk_norm=config.get("_qk_norm", False),
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
            query_pre_attn_scalar=config.get("query_pre_attn_scalar", None),
        )
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=config["intermediate_size"],
            hidden_act=hidden_act,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=config.get("rms_norm_eps", 1e-6), plus_one=rms_plus_one)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.get("rms_norm_eps", 1e-6), plus_one=rms_plus_one)
        if config.get("_has_feedforward_layernorms"):
            self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=config.get("rms_norm_eps", 1e-6), plus_one=rms_plus_one)
            self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=config.get("rms_norm_eps", 1e-6), plus_one=rms_plus_one)
        else:
            self.pre_feedforward_layernorm = None
            self.post_feedforward_layernorm = None

    def forward(self, hidden_states, position_cos, position_sin, kv_cache=None, block_tables=None, slot_idx=None, layer_idx=0, use_cache=False, num_cached_tokens=0, num_cached_tokens_list=None, attn_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_cos, position_sin, kv_cache, block_tables, slot_idx, layer_idx, use_cache=use_cache, num_cached_tokens=num_cached_tokens, num_cached_tokens_list=num_cached_tokens_list, attn_mask=attn_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TransformerModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config["num_hidden_layers"])])
        rms_plus_one = config.get("_rms_norm_plus_one", False)
        self.norm = RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-6), plus_one=rms_plus_one)

        head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        max_pos = config.get("max_position_embeddings", 32768)
        rope_theta = config.get("rope_theta", 10000.0)
        rope_scaling = config.get("rope_scaling", None)
        # Normalize rope_scaling: only pass through if it's a supported type
        if rope_scaling is not None and rope_scaling.get("rope_type") not in ("llama3",):
            rope_scaling = None

        # Per-layer RoPE: support different rope_theta per layer
        layer_rope_thetas = config.get("_layer_rope_thetas", None)
        if layer_rope_thetas is not None:
            # Create one RotaryEmbedding per unique theta
            unique_thetas = dict.fromkeys(layer_rope_thetas)
            self._rope_emb_dict = nn.ModuleDict()
            for theta in unique_thetas:
                key = str(theta).replace(".", "_")
                self._rope_emb_dict[key] = RotaryEmbedding(dim=head_dim, max_position_embeddings=max_pos, base=theta, rope_scaling=rope_scaling)
            self._layer_rope_thetas = layer_rope_thetas
            self.rotary_emb = None
        else:
            self.rotary_emb = RotaryEmbedding(dim=head_dim, max_position_embeddings=max_pos, base=rope_theta, rope_scaling=rope_scaling)
            self._rope_emb_dict = None
            self._layer_rope_thetas = None

        self.num_layers = config["num_hidden_layers"]
        self.head_dim = head_dim
        self.embed_scale = config.get("_embed_scale", None)

    def forward(self, input_ids, positions, kv_cache=None, block_tables=None, slot_idx=None, use_cache=False, num_cached_tokens=0, num_cached_tokens_list=None, attn_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        if self.embed_scale is not None:
            hidden_states = hidden_states * self.embed_scale

        if self._rope_emb_dict is not None:
            # Per-layer RoPE: pre-compute cos/sin for all unique thetas
            rope_cache = {}
            for theta_str, rotary in self._rope_emb_dict.items():
                cos, sin = rotary(positions, hidden_states.device)
                cos = cos[positions]
                sin = sin[positions]
                if cos.dim() == 3:
                    cos = cos.unsqueeze(1)
                    sin = sin.unsqueeze(1)
                else:
                    cos = cos.unsqueeze(0).unsqueeze(0)
                    sin = sin.unsqueeze(0).unsqueeze(0)
                rope_cache[theta_str] = (cos, sin)

            for i, layer in enumerate(self.layers):
                key = str(self._layer_rope_thetas[i]).replace(".", "_")
                cos, sin = rope_cache[key]
                hidden_states = layer(hidden_states, cos, sin, kv_cache, block_tables, slot_idx, i, use_cache=use_cache, num_cached_tokens=num_cached_tokens, num_cached_tokens_list=num_cached_tokens_list, attn_mask=attn_mask)
        else:
            # Single RoPE for all layers
            cos, sin = self.rotary_emb(positions, hidden_states.device)
            cos = cos[positions]
            sin = sin[positions]
            if cos.dim() == 3:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            else:
                cos = cos.unsqueeze(0).unsqueeze(0)
                sin = sin.unsqueeze(0).unsqueeze(0)

            for i, layer in enumerate(self.layers):
                hidden_states = layer(hidden_states, cos, sin, kv_cache, block_tables, slot_idx, i, use_cache=use_cache, num_cached_tokens=num_cached_tokens, num_cached_tokens_list=num_cached_tokens_list, attn_mask=attn_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class ForCausalLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.model = TransformerModel(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        if config.get("tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.config = config

    def forward(self, input_ids, positions, kv_cache=None, block_tables=None, slot_idx=None, use_cache=False, num_cached_tokens=0, num_cached_tokens_list=None, attn_mask=None):
        hidden_states = self.model(input_ids, positions, kv_cache, block_tables, slot_idx, use_cache=use_cache, num_cached_tokens=num_cached_tokens, num_cached_tokens_list=num_cached_tokens_list, attn_mask=attn_mask)
        logits = self.lm_head(hidden_states)
        return logits
