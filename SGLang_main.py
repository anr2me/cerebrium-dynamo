"""
Cerebrium custom-runtime entrypoint for the SGLang backend:
Dynamo frontend + `dynamo.sglang` worker, using SGLang's native
--enable-lmcache flag.

Ported from the Modal `DynamoSGLangLMCache` class. See the module docstring
in the original Modal script for the honest caveat on this backend: LMCache
runs *inside* the SGLang process here — this is NOT a coordinated
Dynamo<->LMCache connector the way the vLLM backend's LMCacheConnectorV1 is.

Deploy with (from the deploy.py wrapper in this repo):
    python deploy.py SGLang.toml

This process is the literal `entrypoint` Cerebrium execs per
[cerebrium.runtime.custom] in SGLang.toml. It must:
  1. download the model into persistent storage if not already present
  2. compile DeepGEMM if not already cached (see note below — this is a
     startup-time step here, unlike Modal's GPU-attached build step)
  3. launch `python -m dynamo.frontend` and `python -m dynamo.sglang`
  4. wait for the frontend's /health to come up, then warm it up
  5. block forever, serving on SGLANG_FRONTEND_PORT (Cerebrium routes
     /health and /ready straight through to this port — see TOML)
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

SGLANG_FRONTEND_PORT = int(os.environ.get("SGLANG_FRONTEND_PORT", "8000"))
SGLANG_SYSTEM_PORT = int(os.environ.get("SGLANG_SYSTEM_PORT", "8081"))

HF_CACHE_PATH = f"{PERSISTENT_STORAGE}/huggingface-cache"
DG_CACHE_PATH = f"{PERSISTENT_STORAGE}/deepgemm-cache"

LMCACHE_MAX_LOCAL_CPU_GB = os.environ.get("LMCACHE_MAX_LOCAL_CPU_GB", "20")

SGLANG_LMCACHE_CONFIG_PATH = "/tmp/lmcache_config_sglang.yaml"
SGLANG_LMCACHE_CONFIG_YAML = f"""\
chunk_size: 256
local_cpu: true
use_layerwise: true
max_local_cpu_size: {LMCACHE_MAX_LOCAL_CPU_GB}
"""


def compile_deep_gemm_if_needed():
    """Startup-time DeepGEMM compile, cached in /persistent-storage.

    On Modal this ran as a GPU-attached build step (run_function(...,
    gpu=GPU)) so it only ever happened once, at image-build time, off the
    request path. Cerebrium's build stage is CPU-only with no equivalent
    hook for arbitrary Python during build, so this now runs here, at
    container startup, on the live billed GPU — but only on the first cold
    start per project: a marker file in DG_CACHE_PATH (which lives on the
    persistent volume, so it survives across container restarts) skips it
    on every subsequent cold start.
    """
    if not int(os.environ.get("SGLANG_ENABLE_JIT_DEEPGEMM", "0")):
        return

    os.makedirs(DG_CACHE_PATH, exist_ok=True)
    marker = os.path.join(DG_CACHE_PATH, ".compiled")
    if os.path.exists(marker):
        return

    subprocess.run(
        f"python3 -m sglang.compile_deep_gemm --model-path {MODEL_NAME} "
        f"--revision {MODEL_REVISION} --tp {N_GPUS}",
        shell=True,
        check=False,
    )
    with open(marker, "w") as f:
        f.write("done")


def main():
    # Static env vars from the Modal image's .env({...}) calls. There's no
    # [cerebrium.environment] table in cerebrium.toml, so these are set
    # process-wide here instead — this also makes them visible to
    # compile_deep_gemm_if_needed()'s subprocess.run() call below, not just
    # the dynamo.sglang worker_env further down.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST", "8.0 8.6 9.0 9.0a 10.0 10.0a 10.3 10.3a 12.0"
    )
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

    get_hf_token_warning_if_missing()

    print(f"[startup] Ensuring model {MODEL_NAME}@{MODEL_REVISION} is cached...")
    download_model_if_needed(MODEL_NAME, MODEL_REVISION, HF_CACHE_PATH)

    print("[startup] Ensuring DeepGEMM is compiled (if enabled)...")
    compile_deep_gemm_if_needed()

    with open(SGLANG_LMCACHE_CONFIG_PATH, "w") as f:
        f.write(SGLANG_LMCACHE_CONFIG_YAML)

    frontend_cmd = [
        "python3",
        "-m",
        "dynamo.frontend",
        "--http-host",
        "0.0.0.0",
        "--http-port",
        f"{SGLANG_FRONTEND_PORT}",
        "--discovery-backend",
        "file",
    ]
    frontend_process = subprocess.Popen(frontend_cmd)

    worker_cmd = [
        "python3",
        "-m",
        "dynamo.sglang",
        "--model-path",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--discovery-backend",
        "file",
        "--tp",
        f"{N_GPUS}",
        "--cuda-graph-max-bs",
        f"{MAX_INPUTS}",
        "--max-running-requests",
        f"{MAX_INPUTS}",
        "--enable-metrics",
        "--enable-lmcache",
    ]
    worker_env = {
        **os.environ,
        "LMCACHE_CONFIG_FILE": SGLANG_LMCACHE_CONFIG_PATH,
        "DYN_SYSTEM_PORT": f"{SGLANG_SYSTEM_PORT}",
        "HF_HUB_CACHE": HF_CACHE_PATH,
    }
    worker_process = subprocess.Popen(worker_cmd, env=worker_env)

    start_crash_watchdog(frontend_process, worker_process)

    print("[startup] Waiting for Dynamo frontend to become healthy...")
    wait_ready(frontend_process, SGLANG_FRONTEND_PORT, timeout=10 * MINUTES)

    print("[startup] Warming up...")
    make_warmup(SGLANG_FRONTEND_PORT, MODEL_NAME)()

    print(f"[startup] Ready. Serving on port {SGLANG_FRONTEND_PORT}.")

    # Block forever, fate-sharing with the two child processes. Cerebrium
    # proxies HTTP traffic to SGLANG_FRONTEND_PORT directly (the frontend
    # process itself, not this script, answers /health, /ready, and
    # /v1/chat/completions) — this script's only remaining job is to stay
    # alive and keep the watchdog threads running.
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
