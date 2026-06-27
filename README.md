# Dynamo + LMCache on Cerebrium.ai

Serverless [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) deployments — SGLang, vLLM, and TensorRT-LLM backends, each paired with [LMCache](https://github.com/LMCache/LMCache) KV-cache offload — running on [Cerebrium](https://www.cerebrium.ai/).

Ported from a [Modal](https://modal.com/) script (`@app.cls` + `modal.Image`) to Cerebrium's `cerebrium.toml` + custom-runtime model. See [What changed vs. Modal](#what-changed-vs-modal) for the architectural differences this port required.

> **Honest caveat, carried over from the source script:** these three backends are not equally "LMCache-integrated." Only the **vLLM** backend uses a real, jointly-documented Dynamo↔LMCache connector (`LMCacheConnectorV1`). The **SGLang** backend runs LMCache *inside* the SGLang process via its native `--enable-lmcache` flag — not a coordinated Dynamo↔LMCache connector. The **TensorRT-LLM** backend uses LMCache's real `KvCacheConnector` integration, but as of writing depends on **unreleased code** on both sides (see [TRTLLM.toml](#trtllmtoml)). Details are in each backend's docstring.

## Repository layout

| File | Purpose |
|---|---|
| `SGLang.toml` | Cerebrium config for the SGLang + native-LMCache backend |
| `vLLM.toml` | Cerebrium config for the vLLM + `LMCacheConnectorV1` backend |
| `TRTLLM.toml` | Cerebrium config for the TensorRT-LLM + `KvCacheConnector` backend |
| `SGLang_main.py` | Container entrypoint for the SGLang backend |
| `vLLM_main.py` | Container entrypoint for the vLLM backend |
| `TRTLLM_main.py` | Container entrypoint for the TensorRT-LLM backend |
| `dynamo_common.py` | Shared lifecycle helpers (warmup, health-wait, crash watchdog, model download) imported by all three `_main.py` files |
| `deploy.py` | CLI wrapper that swaps a chosen `.toml` into `cerebrium.toml`, runs `cerebrium deploy`, then restores whatever `cerebrium.toml` was there before |

Each backend is its own Cerebrium app — there is no single combined deployment. Pick a backend, deploy it, repeat for the others if you want all three running side by side.

## Prerequisites

- A [Cerebrium](https://www.cerebrium.ai/) account and the [Cerebrium CLI](https://docs.cerebrium.ai/cerebrium/getting-started/introduction) installed and logged in (`cerebrium login`)
- A Hugging Face token if you're serving a gated model (the default model in these configs, `Qwen/Qwen3.6-35B-A3B-FP8`, is public)

## Quickstart

1. Clone this repo and `cd` into it.
2. (Optional) Add `HF_TOKEN` under this Cerebrium project's dashboard → **Secrets**.
3. Deploy the backend you want:

   ```bash
   python deploy.py SGLang.toml    # SGLang + native LMCache
   python deploy.py vLLM.toml      # vLLM + LMCacheConnectorV1  (recommended default)
   python deploy.py TRTLLM.toml    # TensorRT-LLM + KvCacheConnector (unreleased deps)
   ```

   `deploy.py` backs up any existing `cerebrium.toml`, copies the chosen file into place, runs `cerebrium deploy -y`, then restores the original `cerebrium.toml` regardless of whether the deploy succeeded.

4. Once deployed, the dashboard prints your app's base URL. It serves an OpenAI-compatible `/v1/chat/completions` endpoint:

   ```bash
   curl https://api.aws.us-east-1.cerebrium.ai/v4/<project-id>/<app-name>/v1/chat/completions \
     -H "Authorization: Bearer <YOUR_CEREBRIUM_JWT>" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen3.6-35B-A3B-FP8",
       "messages": [{"role": "user", "content": "Explain the Singular Value Decomposition."}]
     }'
   ```

## Configuration

There's no `[cerebrium.environment]` table in `cerebrium.toml` — all runtime config lives in the `_main.py` files as `os.environ.get(KEY, default)`, so every backend deploys with zero extra setup. To override a default, set the variable under the Cerebrium dashboard's **Secrets** tab (for `HF_TOKEN`) or its **Environment Variables** panel (for everything else):

| Variable | Default | Notes |
|---|---|---|
| `HF_TOKEN` | — | Only required for gated HF models |
| `MODEL_NAME` | `Qwen/Qwen3.6-35B-A3B-FP8` | |
| `MODEL_REVISION` | `95a723d08a9490559dae23d0cff1d9466213d989` | Ignored by the TRT-LLM backend — see its table below |
| `N_GPUS` | `1` | Tensor-parallel size |
| `MAX_INPUTS` | `1000` | Max concurrent/batched requests |
| `LMCACHE_MAX_LOCAL_CPU_GB` | `20` | CPU RAM budget for offloaded KV blocks |

Backend-specific:

| Variable | Default | Backend |
|---|---|---|
| `SGLANG_FRONTEND_PORT` / `SGLANG_SYSTEM_PORT` | `8000` / `8081` | SGLang |
| `SGLANG_ENABLE_JIT_DEEPGEMM` | `0` | SGLang — set `1` to JIT-compile DeepGEMM on first cold start (Hopper/sm_90+ only) |
| `VLLM_FRONTEND_PORT` / `VLLM_SYSTEM_PORT` | `8001` / `8082` | vLLM |
| `TRTLLM_FRONTEND_PORT` / `TRTLLM_SYSTEM_PORT` | `8002` / `8083` | TensorRT-LLM |

If you change a `*_FRONTEND_PORT`, update the matching `port` in that backend's `[cerebrium.runtime.custom]` table too — they must match.

## What changed vs. Modal

Porting `@app.cls` + `modal.Image` to `cerebrium.toml` + a plain Python entrypoint required a few structural trade-offs:

- **No GPU-attached build step.** Modal pre-downloaded the model and compiled DeepGEMM during image build, off the request path. Cerebrium's build stage is CPU-only with no equivalent hook, so both now run at **container startup** on the live GPU — gated by a marker file in `/persistent-storage` so only the *first* cold start per project pays the cost.
- **No memory/GPU snapshotting.** Modal's `@modal.enter(snap=True/False)` split and the sleep/wake-up memory-occupation dance don't have a Cerebrium equivalent exposed in `cerebrium.toml`, so they were dropped. Every cold start now runs the full startup path once.
- **One shared persistent volume, not three named ones.** Cerebrium gives each *project* a single 50GB `/persistent-storage` volume rather than Modal's multiple named `modal.Volume`s. All three backends point their HF cache (and, for SGLang, DeepGEMM cache) at subfolders of that one volume — a side effect is that if you deploy more than one backend in the same project, they share the downloaded model weights.
- **No `modal.Secret`.** `HF_TOKEN` is read straight from `os.environ` — set it once under the Cerebrium dashboard's Secrets tab and it's injected automatically.
- **Custom runtime instead of decorators.** `@modal.web_server`, `@modal.enter`, and `@modal.exit` become a single long-running Python process (`entrypoint` in `[cerebrium.runtime.custom]`) that launches the Dynamo frontend + worker as subprocesses, waits for `/health`, warms up, and then blocks — Cerebrium proxies traffic straight to the frontend's port.

## Backend notes

### SGLang.toml

LMCache runs inside the SGLang worker process via `--enable-lmcache`. Base image: `lmsysorg/sglang:latest-cu130-runtime`. Builds Rust + several CUDA-13.0-targeted wheels from custom index URLs via `shell_commands` (these can't be expressed in the declarative `[cerebrium.dependencies.pip]` table, which only supports plain version pins).

### vLLM.toml

Uses vLLM's documented `kv_connector` mechanism: `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`. This is the one combination NVIDIA and LMCache jointly maintain — start here if you just want a working Dynamo + LMCache deployment. Base image: `vllm/vllm-openai:latest`.

### TRTLLM.toml

Uses LMCache's `KvCacheConnector` integration in-process mode. **As of this writing this depends on unreleased code on both sides** — [NVIDIA/TensorRT-LLM PR #12626](https://github.com/NVIDIA/TensorRT-LLM) (connector preset registry) isn't in a stable release, and the matching LMCache adapter is only on LMCache's `dev` branch. `shell_commands` installs LMCache from `git+https://github.com/LMCache/LMCache.git@dev` accordingly. Once both ship stably, simplify that to a pinned-version `pip install`. Base image: `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.2.1`. Requires TensorRT-LLM ≥ 1.2.0 and an LMCache build with the `c_ops` extension (verify with `python -c "import lmcache.c_ops"` inside a running container if the worker fails to start).

Also note: unlike the SGLang/vLLM backends, `TRTLLM_main.py` deliberately omits `--revision` from the `dynamo.trtllm` worker command — no public example confirms that flag exists for this backend, and TensorRT-LLM's CLI sometimes uses underscore-style flags that don't mirror vLLM/SGLang's hyphenated ones. Confirm against your installed version's `--help` output before adding it.

## Hardware

All three default to `HOPPER_H100` (1 GPU), 8 vCPU, 32GB RAM, scaling `0 → 1` replica. These are starting points, not tuned values — adjust `[cerebrium.hardware]` and `[cerebrium.scaling]` in each `.toml` once you've watched real build and runtime logs for your model and traffic pattern.

## Troubleshooting

- **First request after deploy is slow.** Expected — the model download and (for SGLang, if `SGLANG_ENABLE_JIT_DEEPGEMM=1`) DeepGEMM compile happen on the first cold start, not at build time. Subsequent cold starts in the same project skip both.
- **Worker fails at startup.** Check `cerebrium logs` for the failing subprocess. The crash-watchdog in `dynamo_common.py` terminates the frontend if the worker dies (and vice versa), so a frontend-side error in the logs often has a worker-side root cause just above it.
- **Gated model fails to download.** Confirm `HF_TOKEN` is set under the project's Secrets tab and that the token has access to the repo.
