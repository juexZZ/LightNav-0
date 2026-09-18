# Navigation asset preparation (CPU only)

This workflow prepares the pinned original LightNav checkpoint and R2R/RxR VLN-CE
annotations. It does not install the inference stack, initialize CUDA, launch Habitat,
download restricted scene reconstructions, or authorize a GPU job.

For completed downloads, split counts, CPU checks and outstanding blockers, see the
[2026-09-18 status report](NAV_ASSET_STATUS_20260918.md).

## Environment and paths

Use an isolated **asset-only** environment. Python 3.12 is sufficient for these scripts;
it is not the project's Python 3.11 inference/training environment or Habitat's Python 3.9
environment. Do not install the LightNav runtime into it or reuse a VSI environment.

```bash
uv venv --python /usr/bin/python3 .venv-assets
uv pip install --python .venv-assets/bin/python -r scripts/requirements-nav-assets.txt
```

The versioned source manifest is `configs/research/nav_assets.json`. It pins the model
revision and records official VLN-CE archive identifiers and expected English evaluation
counts. Put actual assets outside Git, for example under `/root/lightnav_data_v1`:

```text
models/LightNav-0/                  checkpoint and asset_provenance.json
downloads/                         public annotation ZIP archives
data/datasets/                     R2R/RxR train, val_seen, val_unseen JSON.GZ files
data/scene_datasets/mp3d/           authorized MP3D scenes, not supplied by these scripts
audits/                            generated preflight reports
```

## Explicit downloads

```bash
nice -n 15 ionice -c 2 -n 7 .venv-assets/bin/python scripts/prepare_nav_assets.py \
  --data-root /root/lightnav_data_v1 --model --limit-mib 16

nice -n 15 ionice -c 2 -n 7 .venv-assets/bin/python scripts/prepare_nav_assets.py \
  --data-root /root/lightnav_data_v1 --annotations --limit-mib 16
```

Downloads are sequential within a command; run the commands sequentially as well.
Each payload is bandwidth limited. Model `.part` files are promoted only after checking
the pinned Hugging Face revision's LFS SHA-256 or Git blob SHA-1. Completed provenance
also records local SHA-256 hashes. An existing corrupted model file is an error, not
silently overwritten. A missing provenance file means verification has not completed.

Annotation archives have no publisher checksum in the upstream download instructions.
The script records local hashes and checks ZIP CRC during selected extraction, which is
not publisher-authenticated integrity. Only requested split annotation/GT files are
extracted; legacy encoder/text feature files are excluded. It refuses archive traversal
and symlink entries. Dataset access/usage conditions still apply; this script does not
accept agreements on the user's behalf. Auth failures require human action.

## Read-only preflight

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
  .venv-assets/bin/python scripts/preflight_nav_assets.py \
  --data-root /root/lightnav_data_v1 \
  --scenes-dir /path/to/authorized/scene_datasets \
  --splits train val_seen val_unseen \
  --output /root/lightnav_data_v1/audits/assets.json
```

This command uses only the Python standard library, performs no network calls, and never
imports torch or Habitat. It checks pinned model provenance, file sizes, safetensors
headers/payload extents, RVQ dependencies, annotations, English filtering, episode keys,
GT coverage, scene files/GLB headers, and train/val-unseen scene overlap. Add
`--verify-sha256` to reread and hash every model file; the default avoids another full
checkpoint read and relies on download-time publisher verification plus sizes.

Exit `0` means the selected **local asset checks** passed; exit `2` means blocked or
invalid. Neither means that GPU inference or simulation has been validated. The report
always has `gpu_verified=false`. Check every error rather than bypassing the preflight.
The default split is only `val_unseen`; explicitly request all three for training prep.

## HM3D is not the R2R/RxR scene set

The user-supplied Blob directory was inspected read-only on 2026-09-18:

```text
/mnt/blob-data-sigmasystem/juexiao/nav_data/hm3d_download/
  versioned_data/hm3d-0.2/hm3d/train/   800 scene directories
  versioned_data/hm3d-0.2/hm3d/val/     100 scene directories
```

The inventory manifests and sampled `.glb` / `.basis.glb` / `.basis.navmesh` names
identify it as HM3D 0.2. Directory counts are not a full payload-integrity audit.
R2R/RxR episodes reference specific **Matterport3D (MP3D)** scenes; HM3D scene geometry,
IDs, start locations and routes cannot be substituted or fixed by renaming/symlinking.
Keep this HM3D asset untouched for its original project or future ObjectNav work.

The official data source is the [VLN-CE data documentation](https://github.com/jacobkrantz/VLN-CE#data).
Request the user's authorized MP3D location or download access if none is available.

## Tests

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
  .venv-assets/bin/python -m pytest -q tests/test_nav_asset_preflight.py
```

These synthetic tests cover the asset tools, not the LightNav runtime CPU suite. Real
model loading, RVQ inference, Habitat rendering, RGB rollouts and navigation training
remain behind the GPU gate in the [experiment plan](ADAPTIVE_MEMORY_EXPERIMENT_PLAN.md).
