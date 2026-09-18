# Navigation assets: 2026-09-18 status

Status: **model and annotations prepared; MP3D scenes unresolved; GPU untouched.**
This is a CPU asset-readiness report, not an inference, training or benchmark result.

## Downloaded and verified

Local asset root: `/root/lightnav_data_v1` (outside Git).

- Model: `models/LightNav-0`, repository `LightOriginsHQ/LightNav-0`, revision
  `826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`.
- All 18 selected release files passed the pinned revision's publisher LFS SHA-256
  or Git blob SHA-1 check. Their total size is 9,709,007,978 bytes, including tokenizer,
  processor, configs, the complete weight shard and RVQ bundle.
- Weight shard SHA-256:
  `ffc4a925378a881afa761865048eb8d07c55cacf5eaf66548b6641c39f67af18`.
- Completed model receipts: `models/LightNav-0/asset_provenance.json`.
- Both official VLN-CE annotation archives downloaded. Six JSON.GZ files per dataset
  were extracted: episodes and nDTW GT for `train`, `val_seen`, `val_unseen`.
- Annotation receipts: `annotations_provenance.json`. These record local SHA-256 and
  successful selected ZIP extraction/CRC; no upstream publisher archive checksum was supplied.
- Downloads ran sequentially with a 16 MiB/s target cap and low process/I/O priority.
  No restricted MP3D reconstructions were downloaded and no agreements were accepted.

## Actual annotation audit

| Dataset / split | Raw episodes | Selected episodes | Required MP3D scenes |
| --- | ---: | ---: | ---: |
| R2R train | 10,819 | 10,819 | 61 |
| R2R val_seen | 778 | 778 | 53 |
| R2R val_unseen | 1,839 | 1,839 | 11 |
| RxR train | 60,300 | 19,996 | 59 |
| RxR val_seen | 6,746 | 2,255 | 57 |
| RxR val_unseen | 11,006 | 3,669 | 11 |

RxR selection is English guide instructions (`en-US`, `en-IN`). Episode keys,
instruction/start/reference-path presence and selected-episode nDTW GT coverage passed.
The union of required scenes is **72 MP3D scene IDs** across these splits; train and
val_unseen scene sets do not overlap. This is not a claim that the full MP3D release
contains only 72 scenes.

## User-provided Blob data

Inspected read-only:
`/mnt/blob-data-sigmasystem/juexiao/nav_data/hm3d_download`.

- The inventories identify HM3D 0.2: 800 training and 100 validation scene IDs.
- Its 900 inventory IDs have **zero intersection** with the 72 required MP3D IDs.
- Inventory entries and sample filenames were checked; the complete HM3D payload was
  not downloaded, hashed, modified, renamed or symlinked.
- This directory cannot supply the scene geometry referenced by these R2R/RxR episodes.
  An authorized MP3D scene directory or access still needs to be identified. This audit
  does not claim that no MP3D data exists elsewhere in the user's storage.

Full local report:
`/root/lightnav_data_v1/audits/hm3d_comparison_20260918.json`.
It includes the exact required/missing scene paths, split statistics and inventory
comparison. Exit status `2` is expected: its seven errors are six split-level missing
MP3D scene reports plus the explicit HM3D/MP3D mismatch, not model/annotation failures.

## CPU verification and environment

- Asset-only environment: `/root/LightNav-0/.venv-assets`, Python 3.12, isolated from VSI.
  It is not the required Python 3.11 inference/training environment or Python 3.9 Habitat environment.
- 47 selected CPU tests passed: new asset preflight tests, existing RVQ tests and Habitat
  result-merging tests. This is not the complete LightNav runtime test suite.
- Targeted Ruff checks and `git diff --check` passed.
- The actual downloaded RVQ codebooks loaded on CPU; shape checks, finite `(10, 3)`
  decoding and explicit STOP codes `[6, 122, 174]` passed.
- The actual release's SlowFast tier schema passed the existing validator.
- No LLM/ViT weights were loaded into a runtime, no CUDA/Habitat/EGL task ran, and no
  VSI code, environment, process, checkpoint or backup was changed.

## Remaining work

1. Locate/obtain the user's authorized MP3D scenes and rerun preflight using that root.
2. Prepare isolated inference/training and Habitat environments, plus baseline dry-run wrappers.
3. Implement the navigation training-sample pipeline, adaptive-memory interface and memory-only trainer.
4. After explicit GPU release: rendering/model smoke, original R2R/RxR baseline, training
   labels/features, then memory training and closed-loop evaluation.

Large assets and generated audit files remain outside Git. The initial asset set is now
durable in `lightnav-runtime/assets/assets-20260918-v1`: 35 files, 10,511,108,199 bytes,
with a valid completion marker and a separate full SHA-256 verification. See
[data acquisition and durability](DATA_AND_DURABILITY.md) for exact paths and recovery.
The model/Habitat environments and memory interface continue in the next preparation
stage; this report's earlier 47-test count refers specifically to the asset-preparation batch.
