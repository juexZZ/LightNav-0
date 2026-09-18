"""Immutable, independently verified LightNav snapshots on the dedicated Blob mount."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

from prepare_nav_assets import safe_path

DEFAULT_ROOT = Path('/mnt/blob-data-sigmasystem-out/juexiao/lightnav-runtime')
CATEGORIES = ('assets', 'code', 'environments', 'experiments', 'checkpoints')


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_files(source: Path, category: str) -> list[Path]:
    if category == 'code':
        result = subprocess.run(
            ['git', '-C', str(source), 'ls-files', '-co', '--exclude-standard', '-z'],
            check=True, capture_output=True,
        )
        files = sorted({source / name.decode() for name in result.stdout.split(b'\0') if name})
    else:
        entries = sorted(source.rglob('*'))
        if any(path.is_symlink() for path in entries):
            raise ValueError('Snapshot source must not contain symlinks')
        files = [path for path in entries if path.is_file()]
    if not files:
        raise ValueError('Empty snapshots are not allowed')
    for path in files:
        relative = path.relative_to(source)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Not a regular file: {relative}')
        if any(part in {'.git', '.ssh', '.codex', '.cache', '__pycache__'} or part.startswith('.venv') for part in relative.parts):
            raise ValueError(f'Private/runtime path is not a snapshot payload: {relative}')
        if path.name.startswith('.env') or path.name.endswith(('.part', '.tmp', '.pem', '.key')):
            raise ValueError(f'Secret or incomplete file is not a snapshot payload: {relative}')
        safe_path(source, relative.as_posix())
    return files


def check_blob_root(root: Path) -> None:
    if root != DEFAULT_ROOT or root.resolve() != DEFAULT_ROOT:
        raise ValueError(f'Writes are restricted to the dedicated namespace: {DEFAULT_ROOT}')
    result = subprocess.run(
        ['findmnt', '-n', '-o', 'SOURCE,FSTYPE,TARGET', '-T', str(root.parent)],
        check=True, capture_output=True, text=True,
    )
    if 'blobfuse' not in result.stdout or '/mnt/blob-data-sigmasystem-out' not in result.stdout:
        raise ValueError('Expected Blobfuse output mount is absent; refusing a local fallback')


def load_snapshot(snapshot: Path) -> tuple[dict, bytes]:
    marker = json.loads((snapshot / 'COMPLETE.json').read_text())
    raw = (snapshot / 'FILES.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != marker['manifest_sha256']:
        raise ValueError('Completion marker and manifest disagree')
    manifest = json.loads(raw)
    paths = [row['path'] for row in manifest['files']]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Empty or duplicate snapshot paths')
    return manifest, raw


def verify(snapshot: Path) -> dict:
    manifest, _ = load_snapshot(snapshot)
    for row in manifest['files']:
        target = safe_path(snapshot / 'payload', row['path'])
        if target.is_symlink() or not target.is_file() or target.stat().st_size != row['bytes'] or digest(target) != row['sha256']:
            raise ValueError(f'Snapshot payload checksum mismatch: {row["path"]}')
    print(f'VERIFIED {snapshot}: {len(manifest["files"])} files', flush=True)
    return manifest


def copy_limited(source: Path, target: Path, rate_mib: int) -> None:
    started, copied = time.monotonic(), 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as incoming, target.open('wb') as outgoing:
        for chunk in iter(lambda: incoming.read(1024 * 1024), b''):
            outgoing.write(chunk)
            copied += len(chunk)
            delay = copied / (rate_mib * 1024**2) - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)


def publish(source: Path, snapshot: Path, category: str, rate_mib: int) -> dict:
    source = source.resolve()
    files = source_files(source, category)
    manifest = {
        'schema_version': 1, 'category': category,
        'files': [{'path': path.relative_to(source).as_posix(), 'bytes': path.stat().st_size,
                   'sha256': digest(path)} for path in files],
    }
    if category == 'code':
        manifest['git_commit'] = subprocess.check_output(
            ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True,
        ).strip()
        manifest['git_worktree_dirty'] = bool(subprocess.check_output(
            ['git', '-C', str(source), 'status', '--porcelain'], text=True,
        ).strip())
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    if (snapshot / 'COMPLETE.json').exists():
        old, _ = load_snapshot(snapshot)
        if old != manifest:
            raise ValueError('Completed snapshots are immutable; choose a new snapshot ID')
        return verify(snapshot)
    snapshot.mkdir(parents=True, exist_ok=True)
    for row in manifest['files']:
        target = safe_path(snapshot / 'payload', row['path'])
        if not target.is_file() or target.stat().st_size != row['bytes'] or digest(target) != row['sha256']:
            copy_limited(source / row['path'], target, rate_mib)
        if target.stat().st_size != row['bytes'] or digest(target) != row['sha256']:
            raise ValueError(f'Post-copy Blob checksum mismatch: {row["path"]}')
        print(f'BLOB_VERIFIED {row["path"]} {row["bytes"]}', flush=True)
    (snapshot / 'FILES.json').write_bytes(raw)
    if (snapshot / 'FILES.json').read_bytes() != raw:
        raise ValueError('Blob manifest readback failed')
    marker = {'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'files': len(files),
              'bytes': sum(row['bytes'] for row in manifest['files']),
              'verification': 'Every closed Blob payload reread and SHA-256 verified before this marker'}
    (snapshot / 'COMPLETE.json').write_text(json.dumps(marker, indent=2) + '\n')
    load_snapshot(snapshot)
    print(f'COMPLETE {snapshot}', flush=True)
    return manifest


def restore(snapshot: Path, destination: Path, rate_mib: int) -> None:
    manifest = verify(snapshot)
    for row in manifest['files']:
        target = safe_path(destination, row['path'])
        if target.exists():
            if not target.is_file() or digest(target) != row['sha256']:
                raise ValueError(f'Restore refuses to overwrite different local data: {target}')
            continue
        copy_limited(snapshot / 'payload' / row['path'], target, rate_mib)
        if digest(target) != row['sha256']:
            raise ValueError(f'Restored file checksum mismatch: {target}')
    print(f'RESTORED {destination}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['publish', 'verify', 'restore'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--category', choices=CATEGORIES, required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--limit-mib', type=int, choices=range(1, 65), default=32)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.snapshot):
        parser.error('Snapshot ID must be a single safe path component')
    snapshot = args.root / args.category / args.snapshot
    if args.command == 'publish':
        if args.source is None:
            parser.error('publish requires --source')
        check_blob_root(args.root)
        if snapshot.resolve().is_relative_to(args.source.resolve()):
            parser.error('Backup destination must not be inside the source')
        publish(args.source, snapshot, args.category, args.limit_mib)
    elif args.command == 'verify':
        verify(snapshot)
    else:
        if args.destination is None:
            parser.error('restore requires --destination')
        if args.destination.resolve().is_relative_to(args.root.resolve()):
            parser.error('Restore only to local storage, never inside the backup root')
        restore(snapshot, args.destination, args.limit_mib)


if __name__ == '__main__':
    main()
