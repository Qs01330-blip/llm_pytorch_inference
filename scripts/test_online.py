"""Test all models online service, capture server+client output, write markdown report."""
import subprocess
import time
import re
import sys
import io
import os
import requests
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Auto-detect project root: scripts/ is a direct child of the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)
CLIENT_SCRIPT = PROJECT_ROOT / "examples" / "openai_client.py"
PORT = 8000

MODELS = [
    (r"D:\models\LLM-Research\Llama-3.2-1B-Instruct", "Llama-3.2-1B-Instruct"),
    (r"D:\models\LLM-Research\gemma-3-1b-it", "gemma-3-1b-it"),
    (r"D:\models\Qwen\Qwen2-0.5B-Instruct", "Qwen2-0.5B-Instruct"),
    (r"D:\models\Qwen\Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"),
    (r"D:\models\Qwen\Qwen3-0.6B", "Qwen3-0.6B"),
    (r"D:\models\Qwen\Qwen3.5-0.8B", "Qwen3.5-0.8B"),
    (r"D:\models\gongjy\minimind-3-moe", "minimind-3-moe"),
]


def wait_for_server(timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"http://localhost:{PORT}/v1/models", timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False


def run_client(model_name):
    with open(CLIENT_SCRIPT, "r", encoding="utf-8") as f:
        original = f.read()
    patched = re.sub(r'model="[^"]*"', f'model="{model_name}"', original)
    temp_script = PROJECT_ROOT / "scripts" / "_temp_client.py"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(patched)
    try:
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            capture_output=True, text=True, timeout=300,
            cwd=str(PROJECT_ROOT), encoding="utf-8", errors="replace",
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT"
    finally:
        with open(CLIENT_SCRIPT, "w", encoding="utf-8") as f:
            f.write(original)
        temp_script.unlink(missing_ok=True)


def start_server(model_path):
    return subprocess.Popen(
        [sys.executable, "-m", "mini_vllm.server.api_server",
         "--model-path", model_path, "--port", str(PORT)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT), encoding="utf-8", errors="replace",
    )


def stop_server(proc):
    proc.terminate()
    try:
        stdout, _ = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate(timeout=5)
    return stdout or ""


def extract_server_info(log):
    lines = []
    for line in log.split("\n"):
        s = line.strip()
        if not s:
            continue
        if any(kw in s for kw in [
            "Backend", "Model", "Device", "Dtype", "Layers", "Hidden",
            "Attn", "KV", "Head", "RoPE", "Tied", "KVCache", "Auto-detected",
            "ERROR", "WARNING", "Uvicorn", "Started server",
            "Loading tokenizer", "Models:", "Server:",
            "+---", "+===", "|", "Mapped", "flattened"
        ]):
            lines.append(s)
    return "\n".join(lines)


def main():
    results = []

    for i, (model_path, model_name) in enumerate(MODELS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(MODELS)}] Testing: {model_name}")
        print(f"{'='*60}")

        print("  Starting server...")
        proc = start_server(model_path)
        if not wait_for_server(timeout=180):
            print("  FAIL: Server did not start")
            server_log = stop_server(proc)
            results.append({
                "name": model_name, "path": model_path,
                "server_log": server_log, "server_info": "",
                "client_stdout": "", "client_stderr": "SERVER FAILED",
                "status": "FAIL",
            })
            continue

        time.sleep(2)
        print("  Running client...")
        client_stdout, client_stderr = run_client(model_name)

        print("  Stopping server...")
        server_log = stop_server(proc)
        server_info = extract_server_info(server_log)

        has_ns = "=== Non-streaming ===" in client_stdout
        has_s = "=== Streaming ===" in client_stdout
        status = "PASS" if (has_ns and has_s and client_stdout.strip()) else "FAIL"

        results.append({
            "name": model_name, "path": model_path,
            "server_log": server_log, "server_info": server_info,
            "client_stdout": client_stdout, "client_stderr": client_stderr,
            "status": status,
        })
        print(f"  Status: {status}")
        time.sleep(3)

    write_report(results)


def write_report(results):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_path = DOCS_DIR / "online_test_report.md"

    lines = [
        "# Mini-VLLM Online Service Test Report",
        "",
        f"> **Date**: {ts}",
        f"> **Script**: `examples/openai_client.py`",
        f"> **Test Content**: Non-streaming (快速排序) + Streaming (你是谁?)",
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

        lines.append("### Server Startup Log")
        lines.append("")
        if r["server_info"].strip():
            lines.append("```")
            lines.append(r["server_info"].strip())
            lines.append("```")
        else:
            lines.append("*No server info*")
        lines.append("")

        lines.append("### Client Output (openai_client.py)")
        lines.append("")
        if r["client_stdout"].strip():
            output = r["client_stdout"].strip()
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
            lines.append("*No client output*")
        lines.append("")

        if r["client_stderr"].strip():
            lines.append("### Client Stderr")
            lines.append("")
            lines.append("```")
            lines.append(r["client_stderr"].strip()[:3000])
            lines.append("```")
            lines.append("")

        lines.append("<details>")
        lines.append("<summary>Full Server Log (click to expand)</summary>")
        lines.append("")
        lines.append("```")
        lines.append(r["server_log"].strip()[:5000] if r["server_log"].strip() else "*No server log*")
        lines.append("```")
        lines.append("</details>")
        lines.append("")
        lines.append("---")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
