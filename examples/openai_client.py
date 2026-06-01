# examples/openai_client.py
"""Use OpenAI client to call mini-vllm API"""
import sys
import io

# Windows console UTF-8 support for Chinese output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from openai import OpenAI


def main():
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
    )

    # --- Non-streaming ---
    print("=== Non-streaming ===")
    response = client.chat.completions.create(
        model="Llama-3.2-1B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "用Python写一个快速排序"},
        ],
        temperature=0.7,
        max_tokens=256,
    )
    print(response.choices[0].message.content)
    print()

    # --- Streaming ---
    print("=== Streaming ===")
    stream = client.chat.completions.create(
        model="Llama-3.2-1B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁?"},
        ],
        stream=True,
        max_tokens=256,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
