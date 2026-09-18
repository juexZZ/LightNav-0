# Data acquisition and durable LightNav storage

## Dataset decision

R2R/RxR **VLN-CE** requires Matterport3D (MP3D), not HM3D. The user confirmed on
2026-09-18 that no MP3D data had previously been obtained. The existing Blob directory
`/mnt/blob-data-sigmasystem/juexiao/nav_data/hm3d_download` is HM3D 0.2 and remains
untouched. Its 900 scene IDs do not match the 72 MP3D IDs referenced by our selected
train/validation annotations. Do not rename HM3D scenes or change the benchmark.

### MP3D acquisition plan (human access required)

1. Follow the [official Matterport3D access instructions](https://niessner.github.io/Matterport/)
   to obtain authorized access and its official download script. A human must review
   and accept the applicable terms. The agent must not substitute mirrors or accept them.
2. Follow the [VLN-CE scene preparation instructions](https://github.com/jacobkrantz/VLN-CE#data),
   using the official script's Habitat scene package. No restricted MP3D download has
   been performed yet. Inspect that script's actual version/CLI before running it.
3. Stage archives on local SSD under `/root/lightnav_data_v1/downloads/mp3d/`, extract to
   `/root/lightnav_data_v1/data/scene_datasets/mp3d/<scene>/<scene>.glb`, and retain any
   supplied navmeshes. Use sequential limited-I/O operations, never direct Blob training.
4. Run `scripts/preflight_nav_assets.py` with all three splits. The exact 72 required
   paths are in `audits/hm3d_comparison_20260918.json` under `required_mp3d_scenes`.
   Match both scene IDs and GLB validity; file existence alone is insufficient.
5. Publish a **new immutable assets snapshot** after verification. Do not modify the
   already-completed initial snapshot. Keep license/access records private as needed;
   no credentials, agreements with personal details, or scene payloads go to GitHub.

The model and R2R/RxR annotations are already downloaded; see the
[asset status](NAV_ASSET_STATUS_20260918.md) and [preparation commands](NAV_ASSET_PREPARATION.md).
Annotations are not RGB training rollouts. Rendering and teacher/feature generation
remain behind explicit GPU authorization.

## Namespace and layout

Local working root: `/root/lightnav_data_v1`.
Dedicated durable root: **`/mnt/blob-data-sigmasystem-out/juexiao/lightnav-runtime`**.

```text
lightnav-runtime/
  assets/<snapshot>/         downloaded models, annotations, archives and audit receipts
  code/<snapshot>/           Git-visible source/config/docs; excludes .git and environments
  environments/<snapshot>/  dependency locks and setup/CPU-check receipts, not live venvs
  experiments/<snapshot>/   immutable run plans, configs, metrics and completion receipts
  checkpoints/<snapshot>/  completed memory/LLM/optimizer/scheduler checkpoints
```

Each snapshot has `payload/`, `FILES.json` (size/SHA-256 per file) and **`COMPLETE.json`**
(manifest hash). A snapshot without its valid completion marker is **not recoverable
evidence**. The publish command copies files sequentially, closes and independently
rereads each Blob payload for SHA-256, then writes the manifest and completion marker last.
Interrupted incomplete snapshots can be resumed; completed snapshots are immutable.

### Verified initial asset snapshot

On 2026-09-18, `assets/assets-20260918-v1` completed with **35 files / 10,511,108,199 bytes**.
Its manifest SHA-256 is
`703764f5bd828b2fd0e409b7f57abec808059574802f0a179862ed952ffaeed7`.
Both per-file publish readback and a separate full `verify` invocation passed. This
snapshot includes the model, annotation archives, extracted annotation/GT files and
initial audit receipts, but no MP3D scenes, environments or training checkpoints.

This namespace is separate from `spmem-runtime`: never source spmem environment files,
reuse its shared `latest`, change its controllers, or write to its code/checkpoint paths.
The backup CLI refuses publishing anywhere except the dedicated LightNav Blobfuse root
and refuses a missing-mount local-directory fallback. Filesystem approval is still required.

## Backup and recovery commands

No GPU, third-party Python package, account token, or model load is needed:

```bash
nice -n 15 ionice -c 2 -n 7 python3 scripts/backup_lightnav_runtime.py publish \
  --category assets --snapshot assets-20260918-v1 \
  --source /root/lightnav_data_v1 --limit-mib 32

python3 scripts/backup_lightnav_runtime.py verify \
  --category assets --snapshot assets-20260918-v1

python3 scripts/backup_lightnav_runtime.py restore \
  --category assets --snapshot assets-20260918-v1 \
  --destination /root/lightnav_data_v1 --limit-mib 32
```

For source snapshots, use `--category code --source /root/LightNav-0` and a new snapshot
ID identifying the commit. Only Git-visible files are selected, with additional secret,
symlink, environment and incomplete-file checks. Snapshot sources must be stable; do not
backup a training checkpoint or metrics file while it is being written. Restore verifies
the complete remote snapshot first and refuses to overwrite differing local files.

Do not blindly restore over a dirty repository or an active run. Restore code into a
new checkout/staging directory and review the diff. On a replacement node, clone the
research fork, restore the latest **explicitly chosen verified** asset snapshot, recreate
environments from their recipes/locks, and rerun CPU preflight before any GPU request.

## Future backup policy

- All future LightNav assets, code, environment receipts, experiments and checkpoints
  use this namespace, never `spmem-runtime`.
- Back up changed code and environment recipes after each completed development batch;
  back up completed checkpoints/metrics after each experiment milestone.
- Publish receipts must include the snapshot path, manifest hash, file count and byte count.
  Git push alone is not an asset backup, and a Blob copy without verified `COMPLETE.json`
  is not a completed backup.
- This is an explicit milestone workflow, **not a continuously running backup daemon**.
  Do not claim automatic backup coverage. A future daemon must be separately configured
  and must obey the same namespace, secret filtering and completion protocol.
