"""
Shared helpers for the Dynamo + LMCache Cerebrium deployments.

Ported from the original Modal script's module-level helper functions.
This file must be included in the deployment (covered by `include = ["*"]`
in each cerebrium.toml) since SGLang_main.py / vLLM_main.py / TRTLLM_main.py
all import from it.

=============================================================================
WHAT CHANGED VS. THE MODAL VERSION (read this before editing the mains)
=============================================================================
- No @modal.enter(snap=True/False) split. Cerebrium has no exposed
  memory/GPU-snapshot hook in cerebrium.toml — its "GPU snapshotting" is an
  internal platform optimization, not something app code can opt into or
  branch on. So there's only ONE startup path now: every cold start runs
  the full startup() (launch frontend+worker, wait_ready, warmup). The
  sleep()/wake_up() memory-occupation dance existed on Modal specifically
  to make snapshot/restore cheap; without snapshots to restore from, calling
  it would just add a pointless sleep-then-immediately-wake-up round trip,
  so it has been dropped entirely (not ported as dead code).
- No modal.Volume. HF_CACHE_PATH / DG_CACHE_PATH now point at subfolders of
  Cerebrium's single project-wide /persistent-storage volume (50GB, shared
  across ALL apps in the project — see the cerebrium.toml comments).
- No CPU-only `run_function` build step. download_model() (and, for SGLang,
  compile_deep_gemm()) now run at container *startup* instead of at image
  *build* time, gated by a "has this already been done" check against
  /persistent-storage so repeat cold starts in the same project skip the
  work. This means the FIRST cold start after a fresh deploy is much slower
  than on Modal (multi-GB download + compile happen on a live, billed GPU
  container) — there's no Cerebrium equivalent of Modal's CPU-only
  build-time run_function() to do this for free ahead of time.
- No modal.Secret. HF_TOKEN is read directly from os.environ — set it once
  under your Cerebrium project's Secrets tab and it's injected automatically.
"""

import os
import subprocess
import threading
import time

import requests

MINUTES = 60  # seconds

PERSISTENT_STORAGE = "/persistent-storage"


def make_warmup(frontend_port: int, model_name: str):
    def warmup():
        payload = {
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 16,
            "model": model_name,
        }
        for _ in range(3):
            requests.post(
                f"http://127.0.0.1:{frontend_port}/v1/chat/completions",
                json=payload,
                timeout=10,
            ).raise_for_status()

    return warmup


def wait_ready(process: subprocess.Popen, frontend_port: int, timeout: int = 5 * MINUTES):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            check_running(process)
            requests.get(f"http://127.0.0.1:{frontend_port}/health").raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ):
            time.sleep(1)
    raise TimeoutError(f"Dynamo server not ready within timeout of {timeout} seconds")


def check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def start_crash_watchdog(frontend_process: subprocess.Popen, worker_process: subprocess.Popen):
    """Fate-share the frontend and worker processes: if either exits for any
    reason (crash, OOM kill, etc.) while serving traffic, terminate the
    other so the container doesn't keep running half-alive.

    Same rationale as the Modal version: Popen() returns as soon as the
    child is spawned, so a crash that happens after startup() has already
    returned never raises anywhere a try/finally could catch it. Watching
    each process for the container's lifetime via .wait() is what's needed.
    Runs as daemon threads so they never block container shutdown.
    """

    def _watch_and_kill_sibling(to_watch: subprocess.Popen, to_kill: subprocess.Popen):
        to_watch.wait()
        if to_kill.poll() is None:
            to_kill.terminate()

    threading.Thread(
        target=_watch_and_kill_sibling,
        args=(frontend_process, worker_process),
        daemon=True,
    ).start()
    threading.Thread(
        target=_watch_and_kill_sibling,
        args=(worker_process, frontend_process),
        daemon=True,
    ).start()


def download_model_if_needed(model_name: str, model_revision: str, hf_cache_path: str):
    """Runtime (not build-time) model download — see module docstring.

    Cerebrium has no CPU-only build-time hook equivalent to Modal's
    run_function(), so this runs on container startup instead, before the
    GPU worker launches. snapshot_download's own on-disk cache check makes
    repeat calls (e.g. a second class/container in the same project,
    sharing /persistent-storage) fast no-ops once the weights are present.
    """
    from huggingface_hub import snapshot_download

    os.makedirs(hf_cache_path, exist_ok=True)
    snapshot_download(
        model_name,
        revision=model_revision,
        cache_dir=hf_cache_path,
        ignore_patterns=["*.bin", "*.pth", "original/*", "*.gguf"],
        token=os.environ.get("HF_TOKEN") or None,
    )


def get_hf_token_warning_if_missing():
    if not os.environ.get("HF_TOKEN"):
        print(
            "Warning: no HF_TOKEN found in the environment. Set it under this "
            "Cerebrium project's Secrets tab. Public models will still "
            "download (with throttled bandwidth); gated models will fail."
        )
