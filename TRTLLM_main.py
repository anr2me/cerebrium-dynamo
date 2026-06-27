"""
Cerebrium custom-runtime entrypoint for the TensorRT-LLM backend:
Dynamo frontend + `dynamo.trtllm` worker, using LMCache's real
KvCacheConnector integration (in-process mode).

Ported from the Modal `DynamoTRTLLMLMCache` class. As of the original
script's writing this depends on unreleased code on both sides
(NVIDIA/TensorRT-LLM PR #12626 + LMCache's `dev` branch adapter) — see
TRTLLM.toml for the from-source install this requires, and the original
Modal script's section docstring for full details. Once both ship in
stable releases, simplify the from-source pip installs in TRTLLM.toml to
normal pinned-version installs.

Deploy with (from the deploy.py wrapper in this repo):
    python deploy.py TRTLLM.toml
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

TRTLLM_FRONTEND_PORT = int(os.environ.get("TRTLLM_FRONTEND_PORT", "8002"))
TRTLLM_SYSTEM_PORT = int(os.environ.get("TRTLLM_SYSTEM_PORT", "8083"))

HF_CACHE_PATH = f"{PERSISTENT_STORAGE}/huggingface-cache"

LMCACHE_MAX_LOCAL_CPU_GB = os.environ.get("LMCACHE_MAX_LOCAL_CPU_GB", "20")

TRTLLM_LMCACHE_CONFIG_PATH = "/tmp/lmcache_trtllm_config.yaml"
TRTLLM_LMCACHE_CONFIG_YAML = """\
kv_cache_config:
  enable_block_reuse: true
kv_connector_config:
  connector_module: lmcache.integration.tensorrt_llm.tensorrt_adapter
  connector_scheduler_class: LMCacheKvConnectorScheduler
  connector_worker_class: LMCacheKvConnectorWorker
"""


def main():
    # Static env vars from the Modal image's .env({...}) calls. There's no
    # [cerebrium.environment] table in cerebrium.toml, so these are set
    # process-wide here, which also flows into worker_env via **os.environ
    # below.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST", "8.0 8.6 9.0 9.0a 10.0 10.0a 10.3 10.3a 12.0"
    )
    os.environ.setdefault("PYTHONHASHSEED", "0")

    get_hf_token_warning_if_missing()

    print(f"[startup] Ensuring model {MODEL_NAME}@{MODEL_REVISION} is cached...")
    download_model_if_needed(MODEL_NAME, MODEL_REVISION, HF_CACHE_PATH)

    with open(TRTLLM_LMCACHE_CONFIG_PATH, "w") as f:
        f.write(TRTLLM_LMCACHE_CONFIG_YAML)

    frontend_cmd = [
        "python3",
        "-m",
        "dynamo.frontend",
        "--http-host",
        "0.0.0.0",
        "--http-port",
        f"{TRTLLM_FRONTEND_PORT}",
        "--discovery-backend",
        "file",
    ]
    frontend_process = subprocess.Popen(frontend_cmd)

    worker_cmd = [
        "python3",
        "-m",
        "dynamo.trtllm",
        "--model-path",
        MODEL_NAME,
        # NOTE: same caveat as the Modal version — no public dynamo.trtllm
        # example confirms a --revision flag, and TensorRT-LLM's own CLI
        # uses underscore-style flags in places that don't always mirror
        # vLLM/SGLang's hyphenated style. MODEL_REVISION pinning is
        # intentionally omitted here; confirm against your installed
        # TensorRT-LLM/Dynamo version's --help output before adding it.
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--discovery-backend",
        "file",
        "--tensor-parallel-size",
        f"{N_GPUS}",
        "--max-batch-size",
        f"{MAX_INPUTS}",
        "--extra-engine-args",
        TRTLLM_LMCACHE_CONFIG_PATH,
    ]
    worker_env = {
        **os.environ,
        "DYN_SYSTEM_PORT": f"{TRTLLM_SYSTEM_PORT}",
        "HF_HUB_CACHE": HF_CACHE_PATH,
        # LMCache's TRT-LLM adapter reads LMCacheEngineConfig the same way
        # the vLLM adapter does: LMCACHE_CONFIG_FILE for YAML, otherwise
        # individual LMCACHE_* env vars (used here).
        "LMCACHE_CHUNK_SIZE": "256",
        "LMCACHE_LOCAL_CPU": "True",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": LMCACHE_MAX_LOCAL_CPU_GB,
    }
    worker_process = subprocess.Popen(worker_cmd, env=worker_env)

    start_crash_watchdog(frontend_process, worker_process)

    print("[startup] Waiting for Dynamo frontend to become healthy...")
    wait_ready(frontend_process, TRTLLM_FRONTEND_PORT, timeout=10 * MINUTES)

    print("[startup] Warming up...")
    make_warmup(TRTLLM_FRONTEND_PORT, MODEL_NAME)()

    print(f"[startup] Ready. Serving on port {TRTLLM_FRONTEND_PORT}.")

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
