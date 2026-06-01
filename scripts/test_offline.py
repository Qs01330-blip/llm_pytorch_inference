"""Test all models with offline inference, capture output, write markdown report."""
import subprocess
import sys
import time
import re
import os
from pathlib import Path

# Auto-detect project root: scripts/ is a direct child of the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)
CLIENT_SCRIPT = PROJECT_ROOT / "examples" / "offline_inference.py"

MODELS = [
    (r"D:\models\LLM-Research\Llama-3.2-1B-Instruct", "Llama-3.2-1B-Instruct"),
    (r"D:\models\LLM-Research\gemma-3-1b-it", "gemma-3-1b-it"),
    (r"D:\models\Qwen\Qwen2-0.5B-Instruct", "Qwen2-0.5B-Instruct"),
    (r"D:\models\Qwen\Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"),
    (r"D:\models\Qwen\Qwen3-0.6B", "Qwen3-0.6B"),
    (r"D:\models\Qwen\Qwen3.5-0.8B", "Qwen3.5-0.8B"),
    (r"D:\models\gongjy\minimind-3-moe", "minimind-3-moe"),
]


def run_offline_test(model_path, model_name):
    """Run offline_inference.py with the given model, return (stdout, stderr, status)."""
    # Read original script
    with open(CLIENT_SCRIPT, "r", encoding="utf-8") as f:
        original = f.read()

    # Patch: replace model_path and uncomment sys.stdout encoding
    patched = original
    # Replace the model_path line
    def _repl(m):
        return f'model_path=r"{model_path}"'
    patched = re.sub(
        r'model_path=r?"[^"]*"',
        _repl,
        patched,
        count=1
    )
    # Uncomment sys.stdout lines for Windows UTF-8
    patched = patched.replace("# import sys", "import sys")
    patched = patched.replace("# import io", "import io")
    patched = patched.replace(
        "# if sys.platform == \"win32\":",
        "if sys.platform == \"win32\":"
    )
    patched = patched.replace(
        "    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=\"utf-8\")",
        "    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=\"utf-8\")"
    )

    # Write patched script
    temp_script = PROJECT_ROOT / "scripts" / "_temp_offline.py"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(patched)

    try:
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            capture_output=True, text=True, timeout=300,
            cwd=str(PROJECT_ROOT),
            encoding="utf-8", errors="replace",
        )
        stdout = result.stdout
        stderr = result.stderr
        status = "PASS" if (result.returncode == 0 and stdout.strip()) else "FAIL"
    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = "TIMEOUT: offline inference took too long (300s)"
        status = "FAIL"
    except Exception as e:
        stdout = ""
        stderr = f"Error: {e}"
        status = "FAIL"
    finally:
        # Restore original
        with open(CLIENT_SCRIPT, "w", encoding="utf-8") as f:
            f.write(original)
        temp_script.unlink(missing_ok=True)

    return stdout, stderr, status


def extract_server_info(stderr_text):
    """Extract key info lines from stderr."""
    lines = []
    for line in stderr_text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if any(kw in s for kw in [
            "Backend", "Model", "Device", "Dtype", "Layers", "Hidden",
            "Attn", "KV", "Head", "RoPE", "Tied", "KVCache", "Auto-detected",
            "ERROR", "WARNING", "Loading", "Models:", "Mapped", "flattened",
            "+---", "+===", "|"
        ]):
            lines.append(s)
    return "\n".join(lines)


def main():
    results = []
    total = len(MODELS)

    for i, (model_path, model_name) in enumerate(MODELS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] Testing: {model_name}")
        print(f"  Path: {model_path}")
        print(f"{'='*60}")

        stdout, stderr, status = run_offline_test(model_path, model_name)
        server_info = extract_server_info(stderr)

        results.append({
            "name": model_name,
            "path": model_path,
            "stdout": stdout,
            "stderr": stderr,
            "server_info": server_info,
            "status": status,
        })

        print(f"  Status: {status}")
        if stdout.strip():
            preview = stdout.strip()[:120].replace("\n", " ")
            print(f"  Preview: {preview}...")

        time.sleep(2)

    # Write report
    write_report(results)
    print(f"\n\nAll {total} tests completed!")


def write_report(results):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    report_path = DOCS_DIR / "offline_inference_report_cuda.md"

    lines = [
        "# Mini-VLLM Offline Inference Test Report",
        "",
        f"> **Date**: {ts}",
        f"> **Script**: `examples/offline_inference.py`",
        f"> **Test Content**: Non-streaming (who are you?) + Streaming (快速排序算法)",
        f"> **Sampling**: temperature=0.3, top_p=0.9, max_tokens=512",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| # | Model | Path | Status |",
        "|---|-------|------|--------|",
    ]

    for i, r in enumerate(results, 1):
        icon = "PASS" if r["status"] == "PASS" else "FAIL"
        lines.append(f"| {i} | {r['name']} | `{r['path']}` | {icon} |")

    lines.extend(["", "---", ""])

    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. {r['name']}")
        lines.append("")
        lines.append(f"- **Model Path**: `{r['path']}`")
        lines.append(f"- **Status**: **{r['status']}**")
        lines.append("")

        # Server/Engine info from stderr
        lines.append("### Engine Info (stderr)")
        lines.append("")
        if r["server_info"].strip():
            lines.append("```")
            lines.append(r["server_info"].strip())
            lines.append("```")
        else:
            lines.append("*No engine info*")
        lines.append("")

        # Output - split into non-streaming and streaming
        lines.append("### Output")
        lines.append("")
        if r["stdout"].strip():
            output = r["stdout"].strip()
            parts = output.split("=== Streaming ===")
            if len(parts) == 2:
                ns_part = parts[0].replace("=== Non-streaming ===", "").strip()
                s_part = parts[1].strip()
                lines.append("**Non-streaming:**")
                lines.append("")
                lines.append("```")
                lines.append(ns_part)
                lines.append("```")
                lines.append("")
                lines.append("**Streaming:**")
                lines.append("")
                lines.append("```")
                lines.append(s_part)
                lines.append("```")
            else:
                lines.append("```")
                lines.append(output)
                lines.append("```")
        else:
            lines.append("*No output*")
        lines.append("")

        # Stderr (collapsed)
        if r["stderr"].strip():
            lines.append("<details>")
            lines.append("<summary>Full Stderr (click to expand)</summary>")
            lines.append("")
            lines.append("```")
            lines.append(r["stderr"].strip()[:5000])
            lines.append("```")
            lines.append("</details>")
            lines.append("")

        lines.append("---")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
