# tests/test_e2e_real_model.py
"""End-to-end verification: compare mini-vllm output vs HuggingFace transformers."""
import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = r"D:\models\Qwen\Qwen2-0___5B-Instruct"


@pytest.fixture(scope="module")
def hf_model():
    """Load HuggingFace model for reference."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def mini_engine():
    """Load mini-vllm engine."""
    from mini_vllm.engine.engine import LLMEngine
    from mini_vllm.utils.config import EngineConfig
    config = EngineConfig(model_path=MODEL_PATH, device="cpu", dtype="float32")
    engine = LLMEngine(config)
    return engine


class TestModelLoading:
    """Verify mini-vllm loads the real model correctly."""

    def test_model_loads_without_error(self, mini_engine):
        """Engine should initialize without crashing."""
        assert mini_engine.model is not None
        assert mini_engine.tokenizer is not None

    def test_model_type_detected(self, mini_engine):
        """Should detect model_type=qwen2 from config.json."""
        assert mini_engine.model_config.get("model_type") == "qwen2"

    def test_model_dtype(self, mini_engine):
        """Model should be loaded in float32."""
        param_dtype = next(mini_engine.model.parameters()).dtype
        assert param_dtype == torch.float32

    def test_eos_token_detected(self, mini_engine):
        """EOS token should be auto-detected from generation_config.json."""
        # Qwen2-0.5B-Instruct has eos_token_id=151645 in generation_config.json
        assert mini_engine.eos_token_ids is not None
        assert isinstance(mini_engine.eos_token_ids, list)
        assert len(mini_engine.eos_token_ids) > 0
        assert all(isinstance(e, int) for e in mini_engine.eos_token_ids)


class TestLogitComparison:
    """Compare raw logits between mini-vllm and HuggingFace."""

    def test_single_token_logits(self, hf_model, mini_engine):
        """First-token logits should match HuggingFace within tolerance."""
        hf, tokenizer = hf_model
        prompt = "Hello"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # HuggingFace forward
        with torch.no_grad():
            hf_out = hf(input_ids)
            hf_logits = hf_out.logits[0, -1, :]  # [vocab_size]

        # mini-vllm forward (direct model call, same input)
        positions = torch.arange(input_ids.shape[1], dtype=torch.long)
        with torch.no_grad():
            mv_logits = mini_engine.model(
                input_ids, positions,
                kv_cache=None, block_tables=None, slot_idx=None, use_cache=False,
            )
            mv_logits = mv_logits[0, -1, :].float()

        # Compare top-10 tokens
        hf_top10 = torch.topk(hf_logits, 10)
        mv_top10 = torch.topk(mv_logits, 10)

        print(f"HF  top-10 tokens: {hf_top10.indices.tolist()}")
        print(f"MV  top-10 tokens: {mv_top10.indices.tolist()}")
        print(f"HF  top-10 logits: {hf_top10.values.tolist()}")
        print(f"MV  top-10 logits: {mv_top10.values.tolist()}")

        # At least top-5 should match
        hf_top5 = set(hf_top10.indices[:5].tolist())
        mv_top5 = set(mv_top10.indices[:5].tolist())
        overlap = hf_top5 & mv_top5
        assert len(overlap) >= 4, f"Top-5 overlap too low: {overlap} (HF={hf_top5}, MV={mv_top5})"

    def test_logits_mae(self, hf_model, mini_engine):
        """Mean absolute error between logits should be small."""
        hf, tokenizer = hf_model
        prompt = "The capital of France is"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            hf_logits = hf(input_ids).logits[0, -1, :].float()

        positions = torch.arange(input_ids.shape[1], dtype=torch.long)
        with torch.no_grad():
            mv_logits = mini_engine.model(
                input_ids, positions,
                kv_cache=None, block_tables=None, slot_idx=None, use_cache=False,
            )[0, -1, :].float()

        mae = (hf_logits - mv_logits).abs().mean().item()
        print(f"Logit MAE: {mae:.6f}")
        assert mae < 0.05, f"Logit MAE too high: {mae}"


class TestGenerationComparison:
    """Compare generated text between mini-vllm and HuggingFace."""

    def test_greedy_generation_matches(self, hf_model, mini_engine):
        """Greedy decoding (temperature=0) should produce identical output.

        Uses manual HF loop (full recompute each step) for fair comparison,
        since HF's generate() uses internal KV caching with different numerical
        behavior than full recomputation.
        """
        from mini_vllm.utils.config import SamplingParams

        hf, tokenizer = hf_model
        prompt = "Hello, my name is"

        # HuggingFace greedy generation via manual loop (full recompute each step)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        hf_generated = []
        hf_input = input_ids.clone()
        with torch.no_grad():
            for _ in range(20):
                logits = hf(hf_input).logits[0, -1, :].float()
                token_id = torch.argmax(logits).item()
                hf_generated.append(token_id)
                hf_input = torch.cat([hf_input, torch.tensor([[token_id]])], dim=1)
        hf_text = tokenizer.decode(hf_generated, skip_special_tokens=True)

        # mini-vllm generation
        params = SamplingParams(max_tokens=20, temperature=0.0)
        results = mini_engine.generate([prompt], params)
        mv_text = results[0].output_text

        print(f"HF output: '{hf_text}'")
        print(f"MV output: '{mv_text}'")

        # Greedy decoding should produce identical output
        assert hf_text == mv_text, (
            f"Greedy output differs:\n  HF: '{hf_text}'\n  MV: '{mv_text}'"
        )

    def test_multi_prompt_generation(self, mini_engine):
        """Engine should handle multiple prompts in one call."""
        from mini_vllm.utils.config import SamplingParams

        prompts = ["Hello", "The capital of France is"]
        params = SamplingParams(max_tokens=10, temperature=0.0)
        results = mini_engine.generate(prompts, params)

        assert len(results) == 2
        for i, r in enumerate(results):
            assert len(r.output_token_ids) == 10
            assert r.output_text != ""
            print(f"Prompt {i}: '{prompts[i]}' -> '{r.output_text}'")

    def test_generation_respects_max_tokens(self, mini_engine):
        """Should generate exactly max_tokens output tokens."""
        from mini_vllm.utils.config import SamplingParams

        params = SamplingParams(max_tokens=5, temperature=0.0)
        results = mini_engine.generate(["Test"], params)
        assert len(results[0].output_token_ids) == 5
