# mini_vllm/models/gemma3_model.py
# Standalone Gemma 3 transformer implementation.
# Does NOT depend on model.py — all components are self-contained.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Gemma 3 RMSNorm: output = (1 + weight) * normalize(x), weight init to zeros."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * (1.0 + self.weight.float())).to(orig_dtype)


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



def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Gemma3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


# ---------------------------------------------------------------------------
# Attention with QK-norm and per-layer RoPE
# ---------------------------------------------------------------------------

class Gemma3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        query_pre_attn_scalar: float | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self._scale = query_pre_attn_scalar if query_pre_attn_scalar is not None else head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # QK-norm
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

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

        # QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # RoPE
        q, k = _apply_rotary_pos_emb(q, k, position_cos, position_sin)

        # Write to KV Cache
        if kv_cache is not None and block_tables is not None and slot_idx is not None:
            self._write_kv_cache(kv_cache, k, v, block_tables, slot_idx, layer_idx)
            if use_cache:
                if num_cached_tokens_list is not None:
                    k, v = self._read_kv_cache_batch(kv_cache, block_tables, layer_idx, num_cached_tokens_list, hidden_states.device)
                else:
                    k, v = self._read_kv_cache(kv_cache, block_tables, layer_idx)
                    k = k[:num_cached_tokens + 1]
                    v = v[:num_cached_tokens + 1]
                    k = k.unsqueeze(0).expand(batch_size, -1, -1, -1).transpose(1, 2)
                    v = v.unsqueeze(0).expand(batch_size, -1, -1, -1).transpose(1, 2)

        # GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        scale = math.sqrt(self._scale)
        if attn_mask is not None:
            attn_output = F.scaled_dot_product_attention(q, k, v, scale=1.0 / scale, attn_mask=attn_mask)
        elif use_cache:
            attn_output = F.scaled_dot_product_attention(q, k, v, scale=1.0 / scale)
        else:
            attn_output = F.scaled_dot_product_attention(q, k, v, scale=1.0 / scale, is_causal=True)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

    # -- KV cache helpers (identical logic to model.py) --

    def _write_kv_cache(self, kv_cache, k, v, block_tables, slot_idx, layer_idx):
        block_size = kv_cache.shape[3]
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
        k_list, v_list = [], []
        for block_idx in block_tables:
            k_list.append(kv_cache[layer_idx, 0, block_idx])
            v_list.append(kv_cache[layer_idx, 1, block_idx])
        return torch.cat(k_list, dim=0), torch.cat(v_list, dim=0)

    def _read_kv_cache_batch(self, kv_cache, block_tables_list, layer_idx, num_cached_tokens_list, device):
        batch_size = len(block_tables_list)
        max_cached = max(nc + 1 for nc in num_cached_tokens_list)
        k_list, v_list = [], []
        for i in range(batch_size):
            k_seq, v_seq = self._read_kv_cache(kv_cache, block_tables_list[i], layer_idx)
            k_seq = k_seq[:num_cached_tokens_list[i] + 1]
            v_seq = v_seq[:num_cached_tokens_list[i] + 1]
            if k_seq.shape[0] < max_cached:
                pad_size = max_cached - k_seq.shape[0]
                k_seq = torch.cat([k_seq, torch.zeros(pad_size, *k_seq.shape[1:], device=device, dtype=k_seq.dtype)])
                v_seq = torch.cat([v_seq, torch.zeros(pad_size, *v_seq.shape[1:], device=device, dtype=v_seq.dtype)])
            k_list.append(k_seq)
            v_list.append(v_seq)
        k = torch.stack(k_list, dim=0).transpose(1, 2)
        v = torch.stack(v_list, dim=0).transpose(1, 2)
        return k, v


# ---------------------------------------------------------------------------
# Decoder layer — Gemma 3 has pre/post feedforward layernorms
# ---------------------------------------------------------------------------

class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: dict, rms_norm_eps: float):
        super().__init__()
        hidden_size = config["hidden_size"]

        self.self_attn = Gemma3Attention(
            hidden_size=hidden_size,
            num_heads=config["num_attention_heads"],
            num_kv_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
            head_dim=config.get("head_dim", hidden_size // config["num_attention_heads"]),
            rms_norm_eps=rms_norm_eps,
            query_pre_attn_scalar=config.get("query_pre_attn_scalar", None),
        )
        self.mlp = Gemma3MLP(
            hidden_size=hidden_size,
            intermediate_size=config["intermediate_size"],
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states, position_cos, position_sin, kv_cache=None, block_tables=None, slot_idx=None, layer_idx=0, use_cache=False, num_cached_tokens=0, num_cached_tokens_list=None, attn_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_cos, position_sin, kv_cache, block_tables, slot_idx, layer_idx, use_cache=use_cache, num_cached_tokens=num_cached_tokens, num_cached_tokens_list=num_cached_tokens_list, attn_mask=attn_mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Transformer + CausalLM
# ---------------------------------------------------------------------------

class Gemma3TransformerModel(nn.Module):
    """Gemma 3 transformer with per-layer RoPE and embedding scaling."""
    def __init__(self, config: dict):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.register_buffer(
            "embed_scale",
            torch.tensor(math.sqrt(config["hidden_size"]), dtype=torch.float32),
            persistent=False,
        )

        rms_norm_eps = config.get("rms_norm_eps", 1e-6)
        self.layers = nn.ModuleList([
            Gemma3DecoderLayer(config, rms_norm_eps) for _ in range(config["num_hidden_layers"])
        ])
        self.norm = RMSNorm(config["hidden_size"], eps=rms_norm_eps)

        # Per-layer RoPE: every Nth layer uses global rope_theta, others use local
        head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        max_pos = config.get("max_position_embeddings", 32768)
        rope_theta = config.get("rope_theta", 1000000.0)
        rope_local = config.get("rope_local_base_freq", 10000.0)
        pattern = config.get("sliding_window_pattern", 6)
        num_layers = config["num_hidden_layers"]

        self._layer_rope_thetas = []
        for i in range(num_layers):
            if i % pattern == pattern - 1:
                self._layer_rope_thetas.append(rope_theta)
            else:
                self._layer_rope_thetas.append(rope_local)

        # Create one RotaryEmbedding per unique theta
        unique_thetas = dict.fromkeys(self._layer_rope_thetas)
        self._rope_emb_dict = nn.ModuleDict()
        for theta in unique_thetas:
            key = str(theta).replace(".", "_")
            self._rope_emb_dict[key] = RotaryEmbedding(dim=head_dim, max_position_embeddings=max_pos, base=theta)

    def forward(self, input_ids, positions, kv_cache=None, block_tables=None, slot_idx=None, use_cache=False, num_cached_tokens=0, num_cached_tokens_list=None, attn_mask=None):
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale

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

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Gemma3ForCausalLM(nn.Module):
    """Standalone Gemma 3 causal language model.

    This is a self-contained implementation that does NOT inherit from
    model.py's ForCausalLM. All Gemma 3 specific behavior (RMSNorm+1,
    per-layer RoPE, feedforward layernorms, GELU, QK-norm, embed scaling)
    is built in directly.
    """
    def __init__(self, config: dict):
        super().__init__()
        # Store the raw HuggingFace config for weight loading compatibility
        self.config = config
        self.model = Gemma3TransformerModel(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(self, input_ids, positions, kv_cache=None, block_tables=None, slot_idx=None, use_cache=False, num_cached_tokens=0, num_cached_tokens_list=None, attn_mask=None):
        hidden_states = self.model(input_ids, positions, kv_cache, block_tables, slot_idx, use_cache=use_cache, num_cached_tokens=num_cached_tokens, num_cached_tokens_list=num_cached_tokens_list, attn_mask=attn_mask)
        logits = self.lm_head(hidden_states)
        return logits
