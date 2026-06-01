import torch
from mini_vllm.engine.sampler import Sampler
from mini_vllm.utils.config import SamplingParams


def test_greedy_sampling():
    sampler = Sampler()
    logits = torch.tensor([[0.1, 0.9, 0.3, 0.0]])
    params = SamplingParams(temperature=0.0)
    result = sampler.sample(logits, params)
    assert result == [1]  # argmax


def test_temperature_sampling():
    sampler = Sampler()
    torch.manual_seed(42)
    logits = torch.tensor([[1.0, 10.0, 0.1, 0.0]])
    params = SamplingParams(temperature=1.0, top_k=-1, top_p=1.0)
    result = sampler.sample(logits, params)
    assert isinstance(result[0], int)


def test_top_k_sampling():
    sampler = Sampler()
    torch.manual_seed(42)
    logits = torch.tensor([[0.1, 0.9, 0.3, 0.0]])
    params = SamplingParams(temperature=1.0, top_k=2)
    result = sampler.sample(logits, params)
    assert result[0] in [1, 2]  # only from top-2


def test_top_p_sampling():
    sampler = Sampler()
    torch.manual_seed(42)
    logits = torch.tensor([[0.1, 5.0, 0.3, 0.0]])
    params = SamplingParams(temperature=1.0, top_p=0.9)
    result = sampler.sample(logits, params)
    assert isinstance(result[0], int)
