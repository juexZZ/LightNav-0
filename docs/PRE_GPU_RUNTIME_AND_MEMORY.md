# Pre-GPU environments, baseline dry-run and memory interface

These are preparation tools, not permission to run a GPU job. MP3D access remains
unresolved and all GPUs remain reserved for VSI. No model weights or Simulator are
loaded by the setup/import/dry-run commands below.

## Isolated environments

```bash
bash scripts/setup_lightnav_model_env.sh
bash scripts/setup_lightnav_habitat_env.sh
```

The first creates `/root/LightNav-0/.venv` with Python 3.11.14, torch 2.10.0+cu129,
torchvision 0.25.0+cu129, transformers 5.8.0, vLLM 0.19.1 and CUTLASS DSL 4.5.2.
The second uses its own micromamba root `/root/lightnav_tools/mamba` and environment
`/root/envs/lightnav-habitat` (Python 3.9, habitat-sim 0.3.1 headless/bullet,
habitat-lab 0.3.20231024, numpy 1.23.5). No global conda setup or VSI environment is changed.

Habitat uses only `aihabitat` and `conda-forge`, not the Anaconda `defaults` channel.
EGL/GL libraries are installed inside this environment, not as system packages. Merely
importing these libraries does not test real EGL rendering; do not construct a Simulator.
Both environments completed CPU-import checks on 2026-09-18. The earlier Python 3.12
`.venv-assets` remains separate and is not used as the model or Habitat runtime.

Capture full dependency locks and sanitized receipts into a new local directory:

```bash
python3 scripts/capture_lightnav_environments.py \
  --output /root/lightnav_environment_receipts/env-20260918-v1
```

Publish that directory with `backup_lightnav_runtime.py --category environments`.
Receipts contain package versions/import results, never an environment-variable dump,
account tokens or a live venv. The conda explicit export plus pip freezes provide exact
resolved versions; setup constraints alone do not lock every transitive dependency.
On restore, create the conda environment from the explicit file, then install its pip
packages without uninstalling conda-owned libraries. In the model venv, use the recorded
pip freeze with the official cu129 index and install this repo editable. Preserve the
venv interpreter path: resolving its symlink to the base Python loses venv packages.

## B0 baseline dry-run

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/plan_nav_baseline.py \
  --data-root /root/lightnav_data_v1 \
  --scenes-dir /root/lightnav_data_v1/data/scene_datasets \
  --output-root /root/lightnav_baseline_plans/b0-20260918-v1 \
  --gpu-ids 0 --backend vllm_local
```

`--gpu-ids` only records a proposed future assignment; it does not inspect or use any GPU.
This tool **has no execute option**. It writes per-benchmark copies of the upstream YAML,
changing only dataset/scenes paths, plus `plan.json` and `asset_preflight.json`. It keeps
RxR's `_guide` suffix and English languages, the original 500-step/STOP/controller
behavior, split counts and upstream config hashes. It never runs a shell, model or server.

Missing MP3D returns exit `2`, with the complete command plan still available for audit.
Even if local assets pass, `launch_ready=false` and explicit GPU authorization remains
required. R2R and RxR plans run sequentially, not concurrently on the same ports/GPUs.
`--episodes` is a per-shard count as in the original evaluator; `-1` means the full split.
Do not point the plan at the HM3D directory to suppress a missing-MP3D error.

The first B0 can use vLLM; efficiency comparisons with a future HF memory implementation
also require a matched HF B0. This tool does not silently change the original history rules.

## Incremental memory interface

`src/lightnav/memory/stream.py` implements an explicit **contract**, not a learned
adaptive-memory algorithm or an already-integrated LightNav policy. The original
SlowFast engine/serving path is unchanged; no memory CLI flag is exposed as operational.
No private spmem source or VSI-trained checkpoint is copied into the public fork.

- `MemoryObservation`: absolute causal frame pair, absolute timestamp and a variable
  `[tokens, hidden_size]` feature matrix. No fixed 256-token-per-frame assumption.
- `MemoryWriter`: an `nn.Module` contract with `initial_state`, functional `update`
  and `read`. Concrete writers adapt their state to a mapping of named tensors.
- `MemorySession`: per-episode state, exact tubelet deduplication, monotonic history,
  explicit reset, state/readout token budgets, snapshot/restore and explicit detach.
- `MemoryContext`: history tokens and current native features remain separate. The
  current observation must be later than memory; it is not automatically appended.
- Writer parameters can be shared, but state cannot leak between sessions. Updates
  must not mutate their input state; a failed budget check leaves the session uncommitted.
- Snapshots detach/copy tensors to CPU and bind state to writer identity, hidden size
  and episode. Writer identity should include its checkpoint/config fingerprint.
  State snapshots do not replace the writer/optimizer/scheduler training checkpoint.

Concrete integration still needs the actual adaptive writer/readout, current/history
encoding schedule, DeepStack/mRoPE/timestamp/mask handling, HF injection and training
labels. Tubelet deduplication uses frame pairs, not arbitrary grid changes: use one
consistent encoding layout per observation. The caller determines which observation
has become history; overlapping causal frame pairs can share a frame but are distinct
encoding units. Do not re-encode/rewrite an old history under a different layout.

The tests use a deliberately simple **toy writer**, not the research model. They verify
gradients through a frozen dummy backbone, variable token counts, deduplication, causal
rejection, reset/session isolation, budget failure, restore identity and state detachment.
They do not establish navigation quality, bounded real-writer KV memory, or GPU speed.
Real-writer KV size is reported separately via `cache_elements`; token budgets alone
do not bound arbitrary writer implementation storage.

## Verification

On 2026-09-18 the full CPU regression completed with **663 passed, 1 skipped**.
The actual B0 plan at `/root/lightnav_baseline_plans/b0-20260918-v1/plan.json` reports
zero launched processes and two expected blockers: 11 missing MP3D validation scenes
for R2R and 11 for RxR. These two lists overlap; they are not 22 distinct scenes.
Both Python environments passed imports without loading model weights or creating a
Simulator. Targeted Ruff, shell syntax and diff checks also passed.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  make test PYTHON=.venv/bin/python
```

The full CPU suite uses local loopback sockets for fake Habitat/WebSocket servers.
A sandbox that forbids sockets causes permission failures or waits, not a model failure;
request permission for CPU loopback tests rather than starting a real simulator or
patching tests to pass. GPU-marked tests remain excluded. Real GPU/EGL smoke, B0 metrics,
memory training and closed-loop evaluation must wait for MP3D and explicit GPU release.
