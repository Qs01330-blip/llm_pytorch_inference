# mini_vllm/models/gemma3_cuda.py
# Gemma 3 transformer — CUDA accelerated version.
# Uses CUDA ops for RMSNorm, RoPE, and GeGLU when available, falls back to PyTorch automatically.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_vllm.ops.rmsnorm import RMSNorm, fused_add_rmsnorm_plus_one
from mini_vllm.ops.rope import apply_rotary_pos_emb
from mini_vllm.ops.activation import geglu
from mini_vllm.ops.mlp import FusedMLP
from mini_vllm.ops.embedding import Embedding

Gemma3MLP = FusedMLP


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

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

        # QK-norm: uses ops RMSNorm (plus_one=True for Gemma3)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps, plus_one=True)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps, plus_one=True)

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
        q, k = apply_rotary_pos_emb(q, k, position_cos, position_sin)

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

        # GQA: expand KV heads (expand is 0-copy, cheaper than repeat_interleave)
        if self.num_kv_groups > 1:
            # Use k.size(2) instead of seq_len: after KV cache read, T = cache_len, not query_len
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(
                batch_size, self.num_heads, k.size(2), self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(
                batch_size, self.num_heads, v.size(2), self.head_dim)

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

    # -- KV cache helpers --

    def _write_kv_cache(self, kv_cache, k, v, block_tables, slot_idx, layer_idx):
        """Vectorized scatter K,V into paged KV cache."""
        block_size = kv_cache.shape[3]
        is_batch = isinstance(block_tables[0], list)
        seq_len = k.shape[2]
        device = kv_cache.device

        if is_batch:
            batch_size = k.shape[0]
            positions = torch.tensor(slot_idx, device=device).unsqueeze(1) + torch.arange(seq_len, device=device)
            max_blocks = max(len(bt) for bt in block_tables)
            bt_tensor = torch.zeros(batch_size, max_blocks, device=device, dtype=torch.long)
            for b, bt in enumerate(block_tables):
                bt_tensor[b, :len(bt)] = torch.tensor(bt, device=device)
            block_indices = torch.gather(bt_tensor, 1, (positions // block_size).clamp(max=max_blocks-1))
            block_offsets = positions % block_size
            for b in range(batch_size):
                kv_cache[layer_idx, 0, block_indices[b], block_offsets[b]] = k[b].transpose(0, 1)
                kv_cache[layer_idx, 1, block_indices[b], block_offsets[b]] = v[b].transpose(0, 1)
        else:
            positions = slot_idx + torch.arange(seq_len, device=device)
            bt_tensor = torch.tensor(block_tables, device=device)
            block_indices = bt_tensor[positions // block_size]
            block_offsets = positions % block_size
            kv_cache[layer_idx, 0, block_indices, block_offsets] = k[0].transpose(0, 1)
            kv_cache[layer_idx, 1, block_indices, block_offsets] = v[0].transpose(0, 1)

    def _read_kv_cache(self, kv_cache, block_tables, layer_idx):
        bt = torch.tensor(block_tables, device=kv_cache.device)
        k = kv_cache[layer_idx, 0, bt].reshape(-1, *kv_cache.shape[4:])
        v = kv_cache[layer_idx, 1, bt].reshape(-1, *kv_cache.shape[4:])
        return k, v

    def _read_kv_cache_batch(self, kv_cache, block_tables_list, layer_idx, num_cached_tokens_list, device):
        """Vectorized batch read: gather all blocks at once, truncate, pad."""
        batch_size = len(block_tables_list)
        max_cached = max(nc + 1 for nc in num_cached_tokens_list)
        num_kv_heads, head_dim = kv_cache.shape[4], kv_cache.shape[5]
        max_blocks = max(len(bt) for bt in block_tables_list)
        block_size = kv_cache.shape[3]

        bt_tensor = torch.zeros(batch_size, max_blocks, device=device, dtype=torch.long)
        for b, bt in enumerate(block_tables_list):
            bt_tensor[b, :len(bt)] = torch.tensor(bt, device=device)

        k_all = kv_cache[layer_idx, 0, bt_tensor]
        v_all = kv_cache[layer_idx, 1, bt_tensor]

        k_flat = k_all.reshape(batch_size, -1, num_kv_heads, head_dim)
        v_flat = v_all.reshape(batch_size, -1, num_kv_heads, head_dim)

        k_out = torch.zeros(batch_size, max_cached, num_kv_heads, head_dim, device=device, dtype=kv_cache.dtype)
        v_out = torch.zeros_like(k_out)
        for b in range(batch_size):
            n = num_cached_tokens_list[b] + 1
            k_out[b, :n] = k_flat[b, :n]
            v_out[b, :n] = v_flat[b, :n]

        return k_out.transpose(1, 2), v_out.transpose(1, 2)

# 

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
        # All RMSNorm instances use plus_one=True for Gemma3
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, plus_one=True)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, plus_one=True)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, plus_one=True)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, plus_one=True)

    def forward(self, hidden_states, position_cos, position_sin, kv_cache=None, block_tables=None, slot_idx=None, layer_idx=0, use_cache=False, num_cached_tokens=0, num_cached_tokens_list=None, attn_mask=None):
        # ============================================================
        # 【优化前】原始实现：Add 和 RMSNorm 分开计算，多次读写显存
        # ============================================================
        # residual = hidden_states
        # hidden_states = self.input_layernorm(hidden_states)
        # hidden_states = self.self_attn(hidden_states, position_cos, position_sin, kv_cache, block_tables, slot_idx, layer_idx, use_cache=use_cache, num_cached_tokens=num_cached_tokens, num_cached_tokens_list=num_cached_tokens_list, attn_mask=attn_mask)
        # hidden_states = self.post_attention_layernorm(hidden_states)
        # hidden_states = residual + hidden_states
        #
        # residual = hidden_states
        # hidden_states = self.pre_feedforward_layernorm(hidden_states)
        # hidden_states = self.mlp(hidden_states)
        # hidden_states = self.post_feedforward_layernorm(hidden_states)
        # hidden_states = residual + hidden_states

        # ============================================================
        # 【优化后】Fused Add+RMSNorm
        #
        # ⚠️ Gemma3 的 residual add 顺序与标准模型不同：
        #   - 标准模型：先 add residual，再 layernorm → 可融合
        #   - Gemma3：先 layernorm，再 add residual → 无法融合 post_attn/post_ff
        #
        # 因此 Gemma3 融合 pre_feedforward_layernorm + residual add：
        #   原来: residual + hidden_states → pre_feedforward_layernorm
        #   现在: fused_add_rmsnorm_plus_one 一次完成 add + normalize
        # ============================================================
        eps = self.input_layernorm.eps

        # --- Attention block ---
        # 【注意】Gemma3 先 post_attn_norm 后 add residual，无法融合
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_cos, position_sin, kv_cache, block_tables, slot_idx, layer_idx, use_cache=use_cache, num_cached_tokens=num_cached_tokens, num_cached_tokens_list=num_cached_tokens_list, attn_mask=attn_mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # --- MLP block ---
        # 【注意】Gemma3 的 residual add 在 post_ff_ln 之后，无法融合 post_ff_ln。
        # 保持原始实现：先 add residual，再 post_ff_ln。
        # 唯一可优化的是 fused add residual + pre_ff_ln，
        # 但因为后面还需要原始 residual 做最终 add，需要 clone，收益不大。
        # 因此 Gemma3 此处保持原样。
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
        embed_scale = math.sqrt(config["hidden_size"])
        self.embed_tokens = Embedding(config["vocab_size"], config["hidden_size"], scale=embed_scale)

        rms_norm_eps = config.get("rms_norm_eps", 1e-6)
        self.layers = nn.ModuleList([
            Gemma3DecoderLayer(config, rms_norm_eps) for _ in range(config["num_hidden_layers"])
        ])
        self.norm = RMSNorm(config["hidden_size"], eps=rms_norm_eps, plus_one=True)

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
        hidden_states = self.embed_tokens(input_ids)

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
    """Standalone Gemma 3 causal language model — CUDA accelerated.

    Uses ops RMSNorm (plus_one=True) for all normalization layers.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model = Gemma3TransformerModel(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(self, input_ids, positions, kv_cache=None, block_tables=None, slot_idx=None, use_cache=False, num_cached_tokens=0, num_cached_tokens_list=None, attn_mask=None):
        hidden_states = self.model(input_ids, positions, kv_cache, block_tables, slot_idx, use_cache=use_cache, num_cached_tokens=num_cached_tokens, num_cached_tokens_list=num_cached_tokens_list, attn_mask=attn_mask)
        logits = self.lm_head(hidden_states)
        return logits
