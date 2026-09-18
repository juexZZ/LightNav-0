#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
TOOLS="$HOME/lightnav_tools"
PREFIX="$HOME/envs/lightnav-habitat"
export MAMBA_ROOT_PREFIX="$TOOLS/mamba" MAMBA_DOWNLOAD_THREADS=1
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ ! -L "$PREFIX" ]] || { echo 'Refusing a symlinked Habitat environment' >&2; exit 1; }
mkdir -p "$TOOLS"
if [[ ! -f "$TOOLS/micromamba" ]]; then
    curl --fail --location --retry 3 --limit-rate 16M \
        https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64 \
        --output "$TOOLS/micromamba"
fi
printf '366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3  %s\n' "$TOOLS/micromamba" | sha256sum -c -
chmod u+x "$TOOLS/micromamba"
if [[ ! -x "$PREFIX/bin/python" ]]; then
    "$TOOLS/micromamba" create -y -p "$PREFIX" -f configs/research/habitat-environment.yml
fi
"$PREFIX/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 9)'
uv pip install --python "$PREFIX/bin/python" --no-deps 'habitat-lab==0.3.20231024'
uv pip install --python "$PREFIX/bin/python" -e habitat_server \
    'hydra-core==1.3.2' 'omegaconf==2.3.0' 'gym==0.23.0' \
    'opencv-python-headless==4.9.0.80' 'dtw-python==1.5.3' \
    --constraint configs/research/habitat-constraints.txt
"$PREFIX/bin/python" -c 'import habitat, habitat_sim, lightnav_habitat.serve; print("CPU_IMPORT_OK; no Simulator created")'
