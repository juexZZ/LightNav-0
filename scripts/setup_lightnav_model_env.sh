#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ ! -L .venv ]] || { echo 'Refusing a symlinked model environment' >&2; exit 1; }
uv python install 3.11.14
if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.11.14 .venv; fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11)'
uv pip install --python .venv/bin/python 'torch==2.10.0+cu129' 'torchvision==0.25.0+cu129' \
    --index-url https://download.pytorch.org/whl/cu129
uv pip install --python .venv/bin/python -e '.[test,video,habitat,vllm]' \
    --constraint configs/research/model-constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
.venv/bin/python -c 'import torch, lightnav.cli.eval_habitat; assert not torch.cuda.is_initialized(); print("CPU_IMPORT_OK; GPU not verified")'
