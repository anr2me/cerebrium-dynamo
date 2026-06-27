"""
Cerebrium custom-runtime entrypoint for the vLLM backend:
Dynamo frontend + `dynamo.vllm` worker, using the real, jointly-documented
LMCacheConnectorV1 kv_connector.

Ported from the Modal `DynamoVLLMLMCache` class. Per
https://docs.nvidia.com/dynamo/integrations/kv-cache-integrations/lm-cache
this is the one NVIDIA and LMCache both fully document and jointly
maintain — see the original Modal script's module docstring for the
comparison against the SGLang and TensorRT-LLM backends.

Deploy with (from the deploy.py wrapper in this repo):
    python deploy.py vLLM.toml
"""

import os
import subprocess
import sys
import time

from dynamo_common import (
    PERSISTENT_STORAGE,
    check_running,
    download_model_if_needed,
    get_hf_token_warning_if_missing,
    make_warmup,
    start_crash_watchdog,
    wait_ready,
)

MINUTES = 60

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.6-35B-A3B-FP8")
MODEL_REVISION = os.environ.get(
    "MODEL_REVISION", "95a723d08a9490559dae23d0cff1d9466213d989"
)

N_GPUS = int(os.environ.get("N_GPUS", "1"))
MAX_INPUTS = int(os.environ.get("MAX_INPUTS", "1000"))

VLLM_FRONTEND_PORT = int(os.environ.get("VLLM_FRONTEND_PORT", "8001"))
VLLM_SYSTEM_PORT = int(os.environ.get("VLLM_SYSTEM_PORT", "8082"))

HF_CACHE_PATH = f"{PERSISTENT_STORAGE}/huggingface-cache"

LMCACHE_MAX_LOCAL_CPU_GB = os.environ.get("LMCACHE_MAX_LOCAL_CPU_GB", "20")

VLLM_KV_TRANSFER_CONFIG = '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'


def main():
    # Static env vars from the Modal image's .env({...}) calls. There's no
    # [cerebrium.environment] table in cerebrium.toml, so these are set
    # process-wide here, which also flows into worker_env via **os.environ
    # below.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST", "8.0 8.6 9.0 9.0a 10.0 10.0a 10.3 10.3a 12.0"
    )
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

    get_hf_token_warning_if_missing()

    print(f"[startup] Ensuring model {MODEL_NAME}@{MODEL_REVISION} is cached...")
    download_model_if_needed(MODEL_NAME, MODEL_REVISION, HF_CACHE_PATH)

    frontend_cmd = [
        "python3",
        "-m",
        "dynamo.frontend",
        "--http-host",
        "0.0.0.0",
        "--http-port",
        f"{VLLM_FRONTEND_PORT}",
        "--discovery-backend",
        "file",
    ]
    frontend_process = subprocess.Popen(frontend_cmd)

    worker_cmd = [
        "python3",
        "-m",
        "dynamo.vllm",
        "--model",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--discovery-backend",
        "file",
        "--tensor-parallel-size",
        f"{N_GPUS}",
        "--max-num-seqs",
        f"{MAX_INPUTS}",
        "--enable-metrics",
        "--kv-transfer-config",
        VLLM_KV_TRANSFER_CONFIG,
    ]
    worker_env = {
        **os.environ,
        "DYN_SYSTEM_PORT": f"{VLLM_SYSTEM_PORT}",
        "HF_HUB_CACHE": HF_CACHE_PATH,
        # LMCache's own runtime config, read directly by LMCacheConnectorV1
        # — no config file needed for this backend (same as Modal version).
        "LMCACHE_CHUNK_SIZE": "256",
        "LMCACHE_LOCAL_CPU": "True",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": LMCACHE_MAX_LOCAL_CPU_GB,
    }
    worker_process = subprocess.Popen(worker_cmd, env=worker_env)

    start_crash_watchdog(frontend_process, worker_process)

    print("[startup] Waiting for Dynamo frontend to become healthy...")
    wait_ready(frontend_process, VLLM_FRONTEND_PORT, timeout=10 * MINUTES)

    print("[startup] Warming up...")
    make_warmup(VLLM_FRONTEND_PORT, MODEL_NAME)()

    print(f"[startup] Ready. Serving on port {VLLM_FRONTEND_PORT}.")

    try:
        while True:
            check_running(frontend_process)
            check_running(worker_process)
            time.sleep(5)
    except subprocess.CalledProcessError as e:
        print(f"[fatal] A child process exited unexpectedly: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        worker_process.terminate()
        frontend_process.terminate()


if __name__ == "__main__":
    main()
