from mini_vllm.utils.config import EngineConfig, SamplingParams


def test_default_engine_config():
    config = EngineConfig(model_path="/tmp/model")
    assert config.device == "auto"
    assert config.dtype == "auto"
    assert config.block_size == 16
    assert config.max_num_seqs == 256
    assert config.max_num_batched_tokens == 8192
    assert config.tp_size == 1


def test_cpu_config():
    config = EngineConfig(model_path="/tmp/model", device="cpu")
    assert config.get_torch_dtype() == "float32"


def test_sampling_params_defaults():
    params = SamplingParams()
    assert params.temperature == 1.0
    assert params.top_k == -1
    assert params.top_p == 1.0
    assert params.max_tokens == 512
    assert params.stop == []
    assert params.presence_penalty == 0.0


def test_sampling_params_greedy():
    params = SamplingParams(temperature=0.0)
    assert params.temperature == 0.0


def test_sampling_params_eos():
    params = SamplingParams(eos_token_id=151643)
    assert params.eos_token_id == 151643


def test_sampling_params_default_eos():
    params = SamplingParams()
    assert params.eos_token_id == 2
