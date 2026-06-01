import torch
from mini_vllm.engine.kv_cache import KVCacheManager


def test_kv_cache_allocate_and_free():
    cache = KVCacheManager(
        num_blocks=8, block_size=4, num_layers=2, num_heads=2, head_dim=8, device="cpu"
    )
    blocks = cache.allocate(seq_id=0, num_tokens=4)
    assert len(blocks) == 1
    assert 0 in cache.block_tables[0]

    cache.free(seq_id=0)
    assert 0 not in cache.block_tables


def test_kv_cache_multiple_blocks():
    cache = KVCacheManager(
        num_blocks=8, block_size=4, num_layers=2, num_heads=2, head_dim=8, device="cpu"
    )
    blocks = cache.allocate(seq_id=0, num_tokens=9)
    assert len(blocks) == 3  # ceil(9/4) = 3


def test_kv_cache_exhaustion():
    cache = KVCacheManager(
        num_blocks=2, block_size=4, num_layers=2, num_heads=2, head_dim=8, device="cpu"
    )
    cache.allocate(seq_id=0, num_tokens=4)
    blocks = cache.allocate(seq_id=1, num_tokens=8)
    assert blocks is None  # Not enough blocks


def test_kv_cache_append_token():
    cache = KVCacheManager(
        num_blocks=4, block_size=4, num_layers=2, num_heads=2, head_dim=8, device="cpu"
    )
    cache.allocate(seq_id=0, num_tokens=3)
    assert len(cache.block_tables[0]) == 1
    cache.append_token(seq_id=0)
    cache.append_token(seq_id=0)
    assert len(cache.block_tables[0]) == 2  # First block full, new block allocated
