# mini_vllm/models/qwen3_5_model.py
# Standalone Qwen 3.5 transformer implementation with hybrid attention.
# Does NOT depend on model.py — all components are self-contained.
# Supports: Gated Delta Rule linear attention + standard full attention with Q-gate.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Qwen3.5 RMSNorm: output = (1 + weight) * normalize(x), weight init to zeros."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * (1.0 + self.weight.float())).to(orig_dtype)


class RMSNormGated(nn.Module):
    """RMSNorm with gating: norm(x) * sigmoid(gate)."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight.float() * x
        x = x * F.silu(gate.float())
        return x.to(orig_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 262144, base: float = 10000000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, rotary_dim: int):
    """Apply partial RoPE: only first rotary_dim dimensions are rotated."""
    if rotary_dim < q.shape[-1]:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot = q_rot * cos + _rotate_half(q_rot) * sin
        k_rot = k_rot * cos + _rotate_half(k_rot) * sin
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
    else:
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
    return q, k


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# Gated Delta Net — linear attention with recurrent state
# ---------------------------------------------------------------------------

class GatedDeltaNet(nn.Module):
    """Gated Delta Rule linear attention layer for Qwen3.5.

    Supports two modes:
    - Prefill: chunked gated delta rule (matches HF torch_chunk_gated_delta_rule)
    - Decode: recurrent gated delta rule with cached state
    """
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_v_heads = config.get("linear_num_value_heads", 16)
        self.num_k_heads = config.get("linear_num_key_heads", 16)
        self.head_k_dim = config.get("linear_key_head_dim", 128)
        self.head_v_dim = config.get("linear_value_head_dim", 128)
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.get("linear_conv_kernel_dim", 4)
        self.layer_idx = layer_idx

        # QKV projection
        conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = nn.Linear(self.hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # Depthwise conv1d
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, bias=False,
            kernel_size=self.conv_kernel_size,
            groups=conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # Gating parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # Output norm + projection
        self.norm = RMSNormGated(self.head_v_dim, eps=config.get("rms_norm_eps", 1e-6))
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def _chunk_gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 64,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = True,
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Chunked gated delta rule for prefill (matches HF torch_chunk_gated_delta_rule).

        Args:
            query: [B, H, T, k_dim] (float32, raw — L2 norm applied inside)
            key:   [B, H, T, k_dim]
            value: [B, H, T, v_dim]
            g:     [B, H, T] (log-decay gate, already negated+softplussed)
            beta:  [B, H, T] (sigmoid gate)
            chunk_size: 64 (default, matches HF)
            initial_state: [B, H, k_dim, v_dim] or None
            output_final_state: whether to return final recurrent state
            scale: query scale (1/sqrt(k_dim))

        Returns:
            core_attn_out: [B, H, T, v_dim]
            last_recurrent_state: [B, H, k_dim, v_dim] or None
        """
        batch_size, num_heads, sequence_length, k_head_dim = query.shape
        v_head_dim = value.shape[-1]

        # L2 normalize QK (matches HF use_qk_l2norm_in_kernel=True)
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

        # Pad to multiple of chunk_size
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        if pad_size > 0:
            query = F.pad(query, (0, 0, 0, pad_size))
            key = F.pad(key, (0, 0, 0, pad_size))
            value = F.pad(value, (0, 0, 0, pad_size))
            beta = F.pad(beta, (0, pad_size))
            g = F.pad(g, (0, pad_size))

        total_sequence_length = sequence_length + pad_size
        query = query * scale

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)

        # Reshape to chunks: [B, H, num_chunks, chunk_size, dim]
        query, key, value, k_beta, v_beta = [
            x.reshape(batch_size, num_heads, -1, chunk_size, x.shape[-1])
            for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(batch_size, num_heads, -1, chunk_size)
        num_chunks = total_sequence_length // chunk_size

        # Causal mask within chunk
        mask = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
        )

        # Chunk decay: cumsum of g within each chunk, then build decay mask
        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

        # Intra-chunk attention with decay
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

        # Initialize recurrent state
        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
            if initial_state is None
            else initial_state.to(value)
        )

        core_attn_out = torch.zeros_like(value)
        mask_upper = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
        )

        # Process each chunk
        for i in range(num_chunks):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
            v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, i] = attn_inter + attn @ v_new
            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
            )

        if not output_final_state:
            last_recurrent_state = None

        # Remove padding and reshape back
        core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
        core_attn_out = core_attn_out[:, :, :sequence_length]
        return core_attn_out, last_recurrent_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int = 0,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states)  # [B, T, conv_dim]
        z = self.in_proj_z(hidden_states)              # [B, T, value_dim]
        b = self.in_proj_b(hidden_states)              # [B, T, num_v_heads]
        a = self.in_proj_a(hidden_states)              # [B, T, num_v_heads]

        # Conv1d — cast to float32 for cuDNN compatibility (grouped conv1d may not support bf16)
        mixed_qkv = mixed_qkv.transpose(1, 2).contiguous()  # [B, conv_dim, T]
        conv_w = self.conv1d.weight.float()

        # Distinguish decode (single token with cached context) from prefill (multi-token)
        is_decode = seq_len == 1 and conv_state is not None and conv_state.any()

        if is_decode:
            # Decode: prepend cached context, update state BEFORE conv (matches HF)
            conv_input = torch.cat([conv_state, mixed_qkv], dim=-1)
            conv_state.copy_(conv_input[:, :, -(self.conv_kernel_size - 1):])
            mixed_qkv = F.silu(F.conv1d(conv_input.float(), conv_w, None, padding=0, groups=conv_w.shape[0])[:, :, -seq_len:])
        else:
            # Prefill: store pre-conv raw mixed_qkv as state, then causal conv
            if conv_state is not None:
                if seq_len >= self.conv_kernel_size:
                    conv_state.copy_(mixed_qkv[:, :, -(self.conv_kernel_size - 1):])
                else:
                    conv_state.copy_(F.pad(mixed_qkv, (self.conv_kernel_size - 1 - seq_len, 0)))
            mixed_qkv = F.silu(F.conv1d(mixed_qkv.float(), conv_w, None, padding=self.conv_kernel_size - 1, groups=conv_w.shape[0])[:, :, :seq_len])
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, T, conv_dim]

        # Split QKV
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        # GQA: expand key/value heads
        kv_groups = self.num_v_heads // self.num_k_heads
        if kv_groups > 1:
            query = query.repeat_interleave(kv_groups, dim=2)
            key = key.repeat_interleave(kv_groups, dim=2)

        # Gating
        beta = b.sigmoid()  # [B, T, num_v_heads]
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # [B, T, num_v_heads]

        # Process gated delta rule
        scale = 1.0 / (self.head_k_dim ** 0.5)

        if is_decode:
            # Decode: recurrent gated delta rule (single token, with cached state)
            # HF does L2 norm in float32 inside the kernel — do the same here
            state = recurrent_state
            q_t = l2norm(query[:, 0].float(), dim=-1) * scale
            k_t = l2norm(key[:, 0].float(), dim=-1)
            v_t = value[:, 0].float()
            beta_t = beta[:, 0].unsqueeze(-1)
            g_t = g[:, 0].float().exp().unsqueeze(-1).unsqueeze(-1)

            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            o_t = (state * q_t.unsqueeze(-1)).sum(dim=-2)

            core_attn_out = o_t.unsqueeze(1)  # [B, 1, num_v_heads, head_v_dim]
            recurrent_state.copy_(state)
        else:
            # Prefill: chunked gated delta rule (matches HF implementation)
            # Transpose to [B, num_v_heads, T, dim], convert to float32, then L2 norm
            query = query.transpose(1, 2).contiguous().to(torch.float32)
            key = key.transpose(1, 2).contiguous().to(torch.float32)
            value = value.transpose(1, 2).contiguous().to(torch.float32)
            beta = beta.transpose(1, 2).contiguous().to(torch.float32)
            g = g.transpose(1, 2).contiguous().to(torch.float32)

            chunk_size = 64
            # Always use recurrent_state as initial_state when available (for correct state carry-over)
            initial_state = recurrent_state if recurrent_state is not None else None

            core_attn_out, last_recurrent_state = self._chunk_gated_delta_rule(
                query, key, value, g, beta,
                chunk_size=chunk_size,
                initial_state=initial_state,
                output_final_state=True,
                scale=scale,
            )
            if recurrent_state is not None and last_recurrent_state is not None:
                recurrent_state.copy_(last_recurrent_state)

        # Cast to model dtype for downstream layers
        model_dtype = self.out_proj.weight.dtype
        core_attn_out = core_attn_out.to(model_dtype)

        # Norm with gate: core_attn_out is [B, H, T, v_dim], z is [B, T, value_dim]
        # Need both in [B, T, H, v_dim] for norm
        if core_attn_out.dim() == 4 and core_attn_out.shape[1] == self.num_v_heads:
            core_attn_out = core_attn_out.transpose(1, 2).contiguous()  # [B, T, H, v_dim]
        z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)

        # Reshape and project
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, self.value_dim)
        output = self.out_proj(core_attn_out)

        return output, conv_state, recurrent_state


# ---------------------------------------------------------------------------
# Full Attention — with Q-gate and partial RoPE
# ---------------------------------------------------------------------------

class Qwen3_5Attention(nn.Module):
    """Full attention with Q-gate and partial RoPE for Qwen3.5."""
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config.get("num_key_value_heads", self.num_heads)
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5

        # Partial RoPE
        self.partial_rotary_factor = config.get("_partial_rotary_factor", 0.25)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)

        # Q projection outputs 2x: Q + Gate
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK-norm
        self.q_norm = RMSNorm(self.head_dim, eps=config.get("rms_norm_eps", 1e-6))
        self.k_norm = RMSNorm(self.head_dim, eps=config.get("rms_norm_eps", 1e-6))

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

        # Q projection: split into Q and Gate
        q_out = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim * 2)
        query_states, gate = torch.chunk(q_out, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # QK-norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        )
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose to [B, heads, T, dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Partial RoPE
        query_states, key_states = _apply_rotary_pos_emb(
            query_states, key_states, position_cos, position_sin, self.rotary_dim
        )

        # KV Cache write/read
        if kv_cache is not None and block_tables is not None and slot_idx is not None:
            self._write_kv_cache(kv_cache, key_states, value_states, block_tables, slot_idx, layer_idx)
            if use_cache:
                if num_cached_tokens_list is not None:
                    key_states, value_states = self._read_kv_cache_batch(
                        kv_cache, block_tables, layer_idx, num_cached_tokens_list, hidden_states.device
                    )
                else:
                    key_states, value_states = self._read_kv_cache(kv_cache, block_tables, layer_idx)
                    key_states = key_states[:num_cached_tokens + 1]
                    value_states = value_states[:num_cached_tokens + 1]
                    key_states = key_states.unsqueeze(0).expand(batch_size, -1, -1, -1).transpose(1, 2)
                    value_states = value_states.unsqueeze(0).expand(batch_size, -1, -1, -1).transpose(1, 2)

        # GQA: expand KV heads
        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        if attn_mask is not None:
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, scale=self.scaling, attn_mask=attn_mask
            )
        elif use_cache:
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, scale=self.scaling
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, scale=self.scaling, is_causal=True
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Apply Q-gate
        attn_output = attn_output * torch.sigmoid(gate)

        return self.o_proj(attn_output)

    # -- KV cache helpers --

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
# MLP
# ---------------------------------------------------------------------------

class Qwen3_5MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu if hidden_act == "silu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder layer — routes between full_attention and linear_attention
# ---------------------------------------------------------------------------

class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        hidden_size = config["hidden_size"]
        self.layer_type = config["_layer_types"][layer_idx]
        self.layer_idx = layer_idx

        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen3_5Attention(config)

        self.mlp = Qwen3_5MLP(
            hidden_size=hidden_size,
            intermediate_size=config["intermediate_size"],
            hidden_act=config.get("hidden_act", "silu"),
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=config.get("rms_norm_eps", 1e-6))
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.get("rms_norm_eps", 1e-6))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_cos: torch.Tensor | None = None,
        position_sin: torch.Tensor | None = None,
        kv_cache: torch.Tensor | None = None,
        block_tables=None,
        slot_idx=None,
        layer_idx: int = 0,
        use_cache: bool = False,
        num_cached_tokens: int = 0,
        num_cached_tokens_list=None,
        attn_mask=None,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        new_conv_state = None
        new_recurrent_state = None

        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states, new_conv_state, new_recurrent_state = self.linear_attn(
                hidden_states, layer_idx=layer_idx,
                conv_state=conv_state, recurrent_state=recurrent_state,
                use_cache=use_cache,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states, position_cos, position_sin,
                kv_cache, block_tables, slot_idx, layer_idx,
                use_cache=use_cache, num_cached_tokens=num_cached_tokens,
                num_cached_tokens_list=num_cached_tokens_list, attn_mask=attn_mask,
            )

        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_conv_state, new_recurrent_state


# ---------------------------------------------------------------------------
# Transformer + CausalLM
# ---------------------------------------------------------------------------

class Qwen3_5TransformerModel(nn.Module):
    """Qwen 3.5 transformer with hybrid linear + full attention."""
    def __init__(self, config: dict):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(config, i) for i in range(config["num_hidden_layers"])
        ])
        self.norm = RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-6))
        self.num_layers = config["num_hidden_layers"]
        self.head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])

        # RoPE for full attention layers
        partial_rotary_factor = config.get("_partial_rotary_factor", 0.25)
        rotary_dim = int(self.head_dim * partial_rotary_factor)
        rope_theta = config.get("_rope_theta", 10000000.0)
        max_pos = config.get("max_position_embeddings", 262144)
        self.rotary_emb = RotaryEmbedding(dim=rotary_dim, max_position_embeddings=max_pos, base=rope_theta)

        # Cache for linear attention layers: {layer_idx: (conv_state, recurrent_state)}
        self._linear_attn_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_initialized = False

    def _init_linear_attn_cache(self, config: dict, device: torch.device, dtype: torch.dtype):
        """Initialize conv_state and recurrent_state for all linear attention layers."""
        conv_kernel_size = config.get("linear_conv_kernel_dim", 4)
        num_v_heads = config.get("linear_num_value_heads", 16)
        head_k_dim = config.get("linear_key_head_dim", 128)
        head_v_dim = config.get("linear_value_head_dim", 128)
        key_dim = config.get("linear_num_key_heads", 16) * head_k_dim
        conv_dim = key_dim * 2 + num_v_heads * head_v_dim

        self._linear_attn_cache = {}
        for i, layer in enumerate(self.layers):
            if layer.layer_type == "linear_attention":
                conv_state = torch.zeros(1, conv_dim, conv_kernel_size - 1, device=device, dtype=dtype)
                recurrent_state = torch.zeros(1, num_v_heads, head_k_dim, head_v_dim, device=device, dtype=torch.float32)
                self._linear_attn_cache[i] = (conv_state, recurrent_state)
        self._cache_initialized = True

    def reset_cache(self):
        """Reset all linear attention caches to zero and shrink to batch_size=1."""
        for i in self._linear_attn_cache:
            conv_state, recurrent_state = self._linear_attn_cache[i]
            # Shrink back to batch_size=1 to avoid stale data from previous batches
            if conv_state.shape[0] > 1:
                new_conv = torch.zeros_like(conv_state[:1])
                new_rec = torch.zeros_like(recurrent_state[:1])
                self._linear_attn_cache[i] = (new_conv, new_rec)
            else:
                conv_state.zero_()
                recurrent_state.zero_()

    def forward(
        self,
        input_ids,
        positions,
        kv_cache=None,
        block_tables=None,
        slot_idx=None,
        use_cache=False,
        num_cached_tokens=0,
        num_cached_tokens_list=None,
        attn_mask=None,
    ):
        hidden_states = self.embed_tokens(input_ids)

        # RoPE for full attention layers
        max_pos = positions.max().item() + 1
        cos, sin = self.rotary_emb(max_pos, hidden_states.device)
        cos = cos[positions]
        sin = sin[positions]
        if cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        else:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)

        batch_size = input_ids.shape[0]
        for i, layer in enumerate(self.layers):
            conv_state, recurrent_state = None, None
            if i in self._linear_attn_cache:
                conv_state, recurrent_state = self._linear_attn_cache[i]
                # Expand cache to match batch size if needed (batch decode)
                if batch_size > 1 and conv_state.shape[0] < batch_size:
                    conv_state = conv_state.expand(batch_size, -1, -1).contiguous()
                    recurrent_state = recurrent_state.expand(batch_size, -1, -1, -1).contiguous()
                    self._linear_attn_cache[i] = (conv_state, recurrent_state)

            hidden_states, new_conv, new_rec = layer(
                hidden_states, cos, sin,
                kv_cache, block_tables, slot_idx, i,
                use_cache=use_cache, num_cached_tokens=num_cached_tokens,
                num_cached_tokens_list=num_cached_tokens_list, attn_mask=attn_mask,
                conv_state=conv_state, recurrent_state=recurrent_state,
            )

            # Update linear attention cache
            if new_conv is not None and i in self._linear_attn_cache:
                self._linear_attn_cache[i] = (new_conv, new_rec)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3_5ForCausalLM(nn.Module):
    """Standalone Qwen 3.5 causal language model.

    Hybrid architecture: Gated Delta Rule linear attention + standard full attention.
    Linear attention layers use recurrent state (conv_state + recurrent_state).
    Full attention layers use standard paged KV cache via engine.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model = Qwen3_5TransformerModel(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def reset_cache(self):
        """Reset linear attention cache. Must be called between generate() calls."""
        self.model.reset_cache()

    def forward(
        self,
        input_ids,
        positions,
        kv_cache=None,
        block_tables=None,
        slot_idx=None,
        use_cache=False,
        num_cached_tokens=0,
        num_cached_tokens_list=None,
        attn_mask=None,
    ):
        # Initialize linear attention cache on first forward pass
        if not self.model._cache_initialized:
            self.model._init_linear_attn_cache(
                self.config, input_ids.device, next(self.parameters()).dtype
            )

        hidden_states = self.model(
            input_ids, positions,
            kv_cache, block_tables, slot_idx,
            use_cache=use_cache, num_cached_tokens=num_cached_tokens,
            num_cached_tokens_list=num_cached_tokens_list, attn_mask=attn_mask,
        )
        logits = self.lm_head(hidden_states)
        return logits
