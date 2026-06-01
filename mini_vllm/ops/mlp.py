import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from mini_vllm.ops._utils import is_ext_loaded


def _use_cuda(tensor: torch.Tensor) -> bool:
    return is_ext_loaded() and tensor.is_cuda


# activation code → CUDA kernel enum (与 mlp_kernel.cu 中的常量对应)
_ACT_CODE = {
    "silu": 0,
    "gelu": 1,
    "gelu_new": 1,
}


# ============================================================
# 【优化前】原始 MLP：gate_proj 和 up_proj 分别做 2 次 matmul
# ============================================================
# class MLP(nn.Module):
#     def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
#         super().__init__()
#         self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
#         self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
#         self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
#         self.hidden_act = hidden_act
#         self.act_fn = _ACT_FN_REGISTRY[hidden_act]
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.hidden_act == "silu":
#             return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
#         return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# ============================================================
# 【优化后】Fused MLP：合并 gate_proj + up_proj 为 1 次 matmul
# + CUDA kernel 融合 split + activation + multiply
#
# 原始: 3 次 matmul + 1 次 activation kernel
# 现在: 2 次 matmul + 1 次 fused_mlp kernel
# ============================================================


class FusedMLP(nn.Module):
    """Fused MLP: 合并 gate_proj + up_proj + activation 为高效流水线。

    forward 流程:
        gate_up = gate_up_proj(x)     # 1 次 matmul（代替原来 gate_proj + up_proj）
        h = fused_mlp_forward(gate_up) # CUDA kernel: split + act(gate) * up
        out = down_proj(h)             # 1 次 matmul

    权重加载时 gate_proj.weight + up_proj.weight → gate_up_proj.weight。
    """

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self._act_code = _ACT_CODE.get(hidden_act, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)  # 1 次 matmul

        if self._act_code >= 0 and _use_cuda(x):
            # 【优化点】CUDA kernel 融合 split + activation + multiply
            # 直接从 gate_up 拼接张量读取，省去 chunk + contiguous 开销
            h = torch.ops.mini_vllm_ops.fused_mlp_forward(
                gate_up, self.intermediate_size, self._act_code)
        else:
            # CPU fallback（无 CUDA 扩展时）
            gate, up = gate_up.chunk(2, dim=-1)
            gate, up = gate.contiguous(), up.contiguous()
            if self.hidden_act in ("gelu", "gelu_new"):
                h = F.gelu(gate, approximate="tanh") * up
            elif self.hidden_act == "relu":
                h = F.relu(gate) * up
            else:
                h = F.silu(gate) * up

        return self.down_proj(h)


def fuse_mlp_weights(state_dict: dict) -> dict:
    """将 checkpoint 中的 gate_proj.weight + up_proj.weight 合并为 gate_up_proj.weight。

    在 load_state_dict 前调用此函数。

    Args:
        state_dict: 已映射的权重字典

    Returns:
        合并后的权重字典
    """
    to_remove = []
    to_add = {}

    for key in list(state_dict.keys()):
        if not key.endswith(".gate_proj.weight"):
            continue
        prefix = key[: -len(".gate_proj.weight")]
        up_key = prefix + ".up_proj.weight"
        if up_key not in state_dict:
            continue

        gate_w = state_dict[key]
        up_w = state_dict[up_key]
        fused_w = torch.cat([gate_w, up_w], dim=0)  # [2*intermediate, hidden]
        fused_key = prefix + ".gate_up_proj.weight"

        to_add[fused_key] = fused_w
        to_remove.extend([key, up_key])

    for k in to_remove:
        state_dict.pop(k, None)
    state_dict.update(to_add)

    return state_dict
