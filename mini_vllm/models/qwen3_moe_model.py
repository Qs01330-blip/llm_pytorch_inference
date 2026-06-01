# mini_vllm/models/qwen3_moe_model.py
# Standalone Qwen3 MoE (Mixture of Experts) implementation.
# Architecture: standard attention with QK-norm + sparse MoE MLP (top-k routing).
# Each layer has a router that selects k experts from N experts per token.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mini_vllm.models.model import RMSNorm, RotaryEmbedding, Attention


class Qwen3MoeExpert(nn.Module):
    """Single expert MLP: SiLU(gate) * up -> down."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoeSparseMoe(nn.Module):
    """Sparse MoE layer: router + N experts, top-k routing."""
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_experts = config.get("num_experts", 8)
        self.top_k = config.get("num_experts_per_tok", 2)
        self.norm_topk_prob = config.get("norm_topk_prob", True)
        intermediate_size = config.get("moe_intermediate_size", config.get("intermediate_size", 2432))

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            Qwen3MoeExpert(self.hidden_size, intermediate_size)
            for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat = hidden_states.view(-1, hidden_size)  # [B*T, H]

        # Router: softmax over experts
        router_logits = self.gate(flat)  # [B*T, E]
        routing_weights = F.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)  # [B*T, K]

        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Dispatch tokens to experts
        final_hidden = torch.zeros_like(flat)
        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx)  # [B*T, K]
            if not expert_mask.any():
                continue
            token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]
            expert_input = flat[token_indices]
            expert_output = self.experts[expert_idx](expert_input)
            # Accumulate weighted outputs (a token may be routed to this expert via multiple top-k slots)
            for k in range(self.top_k):
                k_mask = expert_mask[token_indices, k]
                if k_mask.any():
                    weights = routing_weights[token_indices, k][k_mask].unsqueeze(-1)
                    final_hidden[token_indices[k_mask]] += weights * expert_output[k_mask]

        return final_hidden.view(batch_size, seq_len, hidden_size)


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        hidden_size = config["hidden_size"]
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_heads=config["num_attention_heads"],
            num_kv_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
            head_dim=config.get("head_dim", hidden_size // config["num_attention_heads"]),
            attention_bias=config.get("attention_bias", False),
            qk_norm=True,
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
        )
        self.mlp = Qwen3MoeSparseMoe(config)
        self.input_layernorm = RMSNorm(hidden_size, eps=config.get("rms_norm_eps", 1e-6))
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.get("rms_norm_eps", 1e-6))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        block_tables=None,
        slot_idx=None,
        layer_idx: int = 0,
        use_cache: bool = False,
        num_cached_tokens: int = 0,
        num_cached_tokens_list=None,
        attn_mask=None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_cos, position_sin,
            kv_cache, block_tables, slot_idx, layer_idx,
            use_cache=use_cache, num_cached_tokens=num_cached_tokens,
            num_cached_tokens_list=num_cached_tokens_list, attn_mask=attn_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3MoeModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(config, i) for i in range(config["num_hidden_layers"])
        ])
        self.norm = RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-6))
        self.num_layers = config["num_hidden_layers"]
        self.head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.get("max_position_embeddings", 32768),
            base=config.get("rope_theta", 1000000.0),
        )

    def forward(
        self, input_ids, positions,
        kv_cache=None, block_tables=None, slot_idx=None,
        use_cache=False, num_cached_tokens=0, num_cached_tokens_list=None, attn_mask=None,
    ):
        hidden_states = self.embed_tokens(input_ids)
        max_pos = positions.max().item() + 1
        cos, sin = self.rotary_emb(max_pos, hidden_states.device)
        cos, sin = cos[positions], sin[positions]
        if cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        else:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, cos, sin,
                kv_cache, block_tables, slot_idx, i,
                use_cache=use_cache, num_cached_tokens=num_cached_tokens,
                num_cached_tokens_list=num_cached_tokens_list, attn_mask=attn_mask,
            )
        return self.norm(hidden_states)


class Qwen3MoeForCausalLM(nn.Module):
    """Qwen3 MoE causal language model with tied embeddings."""
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model = Qwen3MoeModel(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        if config.get("tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self, input_ids, positions,
        kv_cache=None, block_tables=None, slot_idx=None,
        use_cache=False, num_cached_tokens=0, num_cached_tokens_list=None, attn_mask=None,
    ):
        hidden_states = self.model(
            input_ids, positions,
            kv_cache, block_tables, slot_idx,
            use_cache=use_cache, num_cached_tokens=num_cached_tokens,
            num_cached_tokens_list=num_cached_tokens_list, attn_mask=attn_mask,
        )
        return self.lm_head(hidden_states)
