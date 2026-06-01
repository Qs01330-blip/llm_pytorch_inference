# tests/test_engine.py
import torch
import pytest
from mini_vllm.engine.engine import LLMEngine
from mini_vllm.utils.config import EngineConfig, SamplingParams


class MockModel(torch.nn.Module):
    """Mock model for testing"""
    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.head = torch.nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids, positions, kv_cache=None, block_tables=None, slot_idx=None, use_cache=False, num_cached_tokens=0, num_cached_tokens_list=None, attn_mask=None):
        x = self.embed(input_ids)
        return self.head(x)


class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.eos_token_id = 99

    def encode(self, text: str) -> list[int]:
        # Simple: map each character to its ordinal, capped at 98
        return [min(ord(c), 98) for c in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(t) if t < 99 else "" for t in token_ids)


def test_engine_generate():
    """Test basic engine generation"""
    config = EngineConfig(model_path="test", device="cpu", dtype="float32")
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = config
    engine.model = MockModel()
    engine.device = "cpu"
    engine.dtype = "float32"
    engine._torch_dtype = torch.float32
    engine.kv_cache = None

    from mini_vllm.engine.sampler import Sampler
    from mini_vllm.engine.scheduler import Scheduler
    engine.sampler = Sampler()
    engine.scheduler = Scheduler(config)
    engine.tokenizer = MockTokenizer()
    engine.eos_token_ids = [99]

    # Simple test: single request generation
    prompts = ["Hello"]
    params = SamplingParams(max_tokens=3, temperature=0.0)
    results = engine.generate(prompts, params)
    assert len(results) == 1
    assert len(results[0].output_token_ids) == 3
