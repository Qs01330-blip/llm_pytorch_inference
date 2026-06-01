# examples/offline_inference.py
"""Offline inference: non-streaming and streaming examples"""
import sys
import io

# Windows console UTF-8 support for Chinese output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from mini_vllm.engine.engine import LLMEngine
from mini_vllm.utils.config import EngineConfig, SamplingParams


def main():
    config = EngineConfig(
        #model_path=r"D:\models\gongjy\minimind-3-moe",
        #model_path=r"D:\models\Qwen\Qwen2-0.5B-instruct",
        #model_path=r"D:\models\Qwen\Qwen2.5-0.5B-Instruct",
        #model_path=r"D:\models\Qwen\Qwen3-0.6B",
        #model_path=r"D:\models\Qwen\Qwen3.5-0.8B",
        #model_path=r"D:\models\LLM-Research\Llama-3.2-1B-Instruct",
        model_path=r"D:\models\LLM-Research\gemma-3-1b-it",
      

    

        device="auto",
        dtype="float16",
    )

    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=512,
    )

    print("Loading model...")
    engine = LLMEngine(config)
    print("Model loaded.\n")

    # --- Non-streaming ---
    print("=== Non-streaming ===")
    messages = [{"role": "user", "content": "who are you?"}]
    prompt = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    results = engine.generate([prompt], sampling_params)
    print(results[0].output_text)
    print()

    # --- Streaming ---
    print("=== Streaming ===")
    messages = [{"role": "user", "content": "用Python写一个快速排序算法"}]
    prompt = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    for idx, token_text, finished in engine.generate_stream([prompt], sampling_params):
        print(token_text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
