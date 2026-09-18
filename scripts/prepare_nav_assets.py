"""Download pinned public assets without torch, CUDA, or Habitat imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "configs/research/nav_assets.json"


def safe_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe asset path: {name}")
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Asset escapes destination: {name}")
    return target


def hashes(path: Path) -> dict:
    size = path.stat().st_size
    digest = hashlib.sha256()
    git_digest = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            git_digest.update(chunk)
    return {"size_bytes": size, "sha256": digest.hexdigest(), "git_blob_sha1": git_digest.hexdigest()}


def verify_file(path: Path, metadata: dict) -> dict:
    actual = hashes(path)
    if actual["size_bytes"] != metadata["size"]:
        raise ValueError(f"Size mismatch: {path.name}")
    if metadata.get("lfs"):
        valid = actual["sha256"] == metadata["lfs"]["sha256"]
    else:
        valid = actual["git_blob_sha1"] == metadata["blobId"]
    if not valid:
        raise ValueError(f"Publisher checksum mismatch: {path.name}")
    return actual


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def download_model(manifest: dict, root: Path, limit_mib: int) -> None:
    model = manifest["model"]
    repository, revision = model["repo_id"], model["revision"]
    api_url = f"https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true"
    result = subprocess.run(
        ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "60", api_url],
        check=True, capture_output=True, text=True,
    )
    metadata = json.loads(result.stdout)
    if metadata["sha"] != revision:
        raise ValueError("Model API revision differs from the pinned revision")
    entries = {entry["rfilename"]: entry for entry in metadata["siblings"]}
    model_root = root / "models/LightNav-0"
    model_root.mkdir(parents=True, exist_ok=True)
    required_bytes = sum(entries[name]["size"] for name in model["files"])
    if shutil.disk_usage(model_root).free < required_bytes + 1024**3:
        raise ValueError("Insufficient free space for checkpoint plus 1 GiB headroom")
    provenance = {"repo_id": repository, "revision": revision, "files": {}}
    for name in model["files"]:
        target = safe_path(model_root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            partial = target.with_name(target.name + ".part")
            print(f"Downloading {name} (one stream, <= {limit_mib} MiB/s)", flush=True)
            if not partial.exists() or partial.stat().st_size != entries[name]["size"]:
                subprocess.run(
                    ["curl", "--fail", "--silent", "--show-error", "--location",
                     "--connect-timeout", "30", "--retry", "3", "--retry-delay", "2",
                     "--speed-time", "60", "--speed-limit", "1024",
                     "--limit-rate", f"{limit_mib}M", "--continue-at", "-",
                     "--output", str(partial),
                     f"https://huggingface.co/{repository}/resolve/{revision}/{name}"],
                    check=True,
                )
            verified = verify_file(partial, entries[name])
            partial.replace(target)
        else:
            verified = verify_file(target, entries[name])
        provenance["files"][name] = verified
        print(f"Verified {name}", flush=True)
    provenance["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(model_root / "asset_provenance.json", provenance)


def extract_annotations(archive: Path, target_root: Path, dataset: dict, splits: list[str]) -> list[Path]:
    expected = {
        f"{dataset['directory']}/{split}/{dataset[field].format(split=split)}"
        for split in splits for field in ("episodes", "ground_truth")
    }
    extracted = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in member.filename:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Archive symlink is not allowed: {member.filename}")
            if member.filename not in expected:
                continue
            target = safe_path(target_root, member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            with bundle.open(member) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            temporary.replace(target)
            extracted.append(target)
    if not extracted:
        raise ValueError(f"No expected annotation paths found in {archive.name}")
    return extracted


def download_annotations(manifest: dict, root: Path, limit_mib: int) -> None:
    import gdown

    archive_root = root / "downloads"
    archive_root.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name, dataset in manifest["datasets"].items():
        archive = archive_root / f"{dataset['directory']}.zip"
        if not archive.exists():
            partial = archive.with_name(archive.name + ".part")
            result = gdown.download(
                id=dataset["google_drive_id"], output=str(partial), use_cookies=False,
                speed=limit_mib * 1024**2, resume=True,
            )
            if result is None or not zipfile.is_zipfile(partial):
                raise ValueError(f"Download did not produce a ZIP archive: {name}")
            partial.replace(archive)
        extracted = extract_annotations(
            archive, root / "data/datasets", dataset, manifest["allowed_splits"],
        )
        provenance[name] = {
            "google_drive_id": dataset["google_drive_id"], "archive": hashes(archive),
            "verification": "local SHA-256 and extracted ZIP CRC; no publisher checksum supplied",
            "files": {str(path.relative_to(root)): hashes(path) for path in extracted},
        }
        print(f"Extracted {len(extracted)} annotation files for {name}", flush=True)
    write_json(root / "annotations_provenance.json", provenance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--annotations", action="store_true")
    parser.add_argument("--limit-mib", type=int, default=16, choices=range(1, 65))
    args = parser.parse_args()
    if not args.model and not args.annotations:
        parser.error("Explicitly select --model and/or --annotations; nothing downloads by default")
    manifest = json.loads(args.manifest.read_text())
    args.data_root.mkdir(parents=True, exist_ok=True)
    if args.annotations:
        download_annotations(manifest, args.data_root, args.limit_mib)
    if args.model:
        download_model(manifest, args.data_root, args.limit_mib)


if __name__ == "__main__":
    main()
