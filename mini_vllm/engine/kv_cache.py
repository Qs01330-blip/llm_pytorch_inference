import torch
from collections import deque
from mini_vllm.utils.logger import logger


class KVCacheManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        # Pre-allocate KV Cache: [num_layers, 2(K+V), num_blocks, block_size, num_heads, head_dim]
        self.kv_cache = torch.zeros(
            (num_layers, 2, num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        self.free_blocks: deque[int] = deque(range(num_blocks))
        self.block_tables: dict[int, list[int]] = {}
        self.seq_num_tokens: dict[int, int] = {}

        logger.info(
            f"KVCache initialized: {num_blocks} blocks, block_size={block_size}, "
            f"layers={num_layers}, heads={num_heads}, head_dim={head_dim}"
        )

    def allocate(self, seq_id: int, num_tokens: int) -> list[int] | None:
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks_needed:
            return None

        blocks = []
        for _ in range(num_blocks_needed):
            blocks.append(self.free_blocks.popleft())

        self.block_tables[seq_id] = blocks
        self.seq_num_tokens[seq_id] = num_tokens
        return blocks

    def free(self, seq_id: int) -> None:
        if seq_id in self.block_tables:
            blocks = self.block_tables.pop(seq_id)
            self.free_blocks.extend(blocks)
            self.seq_num_tokens.pop(seq_id, None)

    def append_token(self, seq_id: int) -> bool:
        """Append one token, allocating a new block if needed. Returns success."""
        if seq_id not in self.block_tables:
            return False

        num_tokens = self.seq_num_tokens[seq_id] + 1
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if num_blocks_needed > len(self.block_tables[seq_id]):
            if not self.free_blocks:
                return False
            self.block_tables[seq_id].append(self.free_blocks.popleft())

        # Only increment after successful block allocation
        self.seq_num_tokens[seq_id] = num_tokens
        return True

    def get_block_table(self, seq_id: int) -> list[int]:
        return self.block_tables.get(seq_id, [])

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)
