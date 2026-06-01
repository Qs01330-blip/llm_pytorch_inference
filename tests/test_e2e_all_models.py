# tests/test_e2e_all_models.py
"""End-to-end tests for all supported model families."""
import sys
import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fix Windows console encoding for non-ASCII output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# Model paths
MODELS = {
    "qwen2": r"D:\models\Qwen\Qwen2-0___5B-Instruct",
    "qwen2.5": r"D:\models\Qwen\Qwen2___5-0___5B-Instruct",
    "qwen3": r"D:\models\Qwen\Qwen3-0___6B",
    "llama": r"D:\models\LLM-Research\Llama-3___2-1B-Instruct",
    "gemma3": r"D:\models\LLM-Research\gemma-3-1b-it",
}

PROMPT = "Hello, my name is"
MAX_TOKENS = 20


def run_e2e_test(model_path: str):
    """Run e2e test for a single model: compare engine output vs HF manual loop."""
    from mini_vllm.engine.engine import LLMEngine
    from mini_vllm.utils.config import EngineConfig, SamplingParams

    # Load HF model for reference
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    hf_model.eval()

    # Load mini-vllm engine
    config = EngineConfig(model_path=model_path, device="cpu", dtype="float32")
    engine = LLMEngine(config)

    # HF manual greedy loop (full recompute each step)
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt")
    hf_generated = []
    hf_input = input_ids.clone()
    with torch.no_grad():
        for _ in range(MAX_TOKENS):
            logits = hf_model(hf_input).logits[0, -1, :].float()
            token_id = torch.argmax(logits).item()
            hf_generated.append(token_id)
            hf_input = torch.cat([hf_input, torch.tensor([[token_id]])], dim=1)
    hf_text = tokenizer.decode(hf_generated, skip_special_tokens=True)

    # mini-vllm generation
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    results = engine.generate([PROMPT], params)
    mv_text = results[0].output_text
    mv_tokens = results[0].output_token_ids

    print(f"\n{'='*60}")
    print(f"Model: {model_path}")
    print(f"HF  text: '{hf_text}'")
    print(f"MV  text: '{mv_text}'")
    print(f"HF  tokens: {hf_generated}")
    print(f"MV  tokens: {mv_tokens}")

    # Find first divergence
    min_len = min(len(hf_generated), len(mv_tokens))
    first_diff = None
    for i in range(min_len):
        if hf_generated[i] != mv_tokens[i]:
            first_diff = i
            break

    if first_diff is not None:
        print(f"First divergence at token {first_diff}: "
              f"HF={hf_generated[i]}('{tokenizer.decode([hf_generated[i]])}') "
              f"MV={mv_tokens[i]}('{tokenizer.decode([mv_tokens[i]])}')")
    else:
        print(f"All {min_len} tokens match!")

    # Cleanup
    del hf_model, engine

    return hf_text, mv_text, first_diff


@pytest.mark.parametrize("model_name", list(MODELS.keys()))
def test_model_e2e(model_name):
    """Each model family should produce output matching HuggingFace."""
    model_path = MODELS[model_name]
    hf_text, mv_text, first_diff = run_e2e_test(model_path)

    assert hf_text == mv_text, (
        f"[{model_name}] Greedy output differs at token {first_diff}:\n"
        f"  HF: '{hf_text}'\n  MV: '{mv_text}'"
    )


if __name__ == "__main__":
    for name in MODELS:
        try:
            hf_text, mv_text, first_diff = run_e2e_test(MODELS[name])
            match = "PASS" if hf_text == mv_text else f"FAIL (diff at token {first_diff})"
            print(f"Result: {match}")
        except Exception as e:
            print(f"Result: ERROR - {e}")
