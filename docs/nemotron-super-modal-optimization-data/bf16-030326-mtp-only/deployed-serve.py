"""Snapshot-matched BF16 030326 candidate: MTP enabled, Mamba APC disabled."""

import json
import shlex
import subprocess

import modal


GPU_TYPE = "B200"
N_GPU = 2
MODEL_CKPT = "/model/ea_final_nvidia_nemotron_3_super_120b_a12b_bf16_030326_vv0.1"
SERVED_MODEL_NAME = "nemotron-3-super-120b"

app = modal.App("nemotron-super-bf16-030326-mtp-only")
model_volume = modal.Volume.from_name("nemotron-super-weights")
vllm_cache_volume = modal.Volume.from_name(
    "vllm-cache-super-bf16-030326-mtp-only-v25", create_if_missing=True
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("vllm==0.25.1", "huggingface-hub")
    .add_local_file(
        "super_v3_reasoning_parser.py", "/app/super_v3_reasoning_parser.py"
    )
    .add_local_file("chat_template_nano.jinja", "/app/chat_template_nano.jinja")
)


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    volumes={
        "/model": model_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
    scaledown_window=15 * 60,
    timeout=40 * 60,
    max_containers=1,
)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=8000, startup_timeout=35 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL_CKPT,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--served-model-name", SERVED_MODEL_NAME,
        "--async-scheduling",
        "--dtype", "auto",
        "--kv-cache-dtype", "fp8",
        "--tensor-parallel-size", str(N_GPU),
        "--pipeline-parallel-size", "1",
        "--data-parallel-size", "1",
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.9",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--speculative-config", json.dumps(
            {"method": "mtp", "num_speculative_tokens": 1}
        ),
        "--mamba-ssm-cache-dtype", "float16",
        "--max-num-seqs", "512",
        "--chat-template", "/app/chat_template_nano.jinja",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser-plugin", "/app/super_v3_reasoning_parser.py",
        "--reasoning-parser", "super_v3",
        "--kernel-config", json.dumps({"enable_flashinfer_autotune": False}),
    ]
    print(f"Starting vLLM: {shlex.join(cmd)}")
    subprocess.Popen(cmd)
