from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_script('prepare_nav_assets')
backup = load_script('backup_lightnav_runtime')


def test_publish_verify_restore_and_immutable(tmp_path):
    source, snapshot, destination = tmp_path / 'source', tmp_path / 'snapshot', tmp_path / 'restore'
    source.mkdir()
    (source / 'asset').write_bytes(b'payload')
    backup.publish(source, snapshot, 'assets', 64)
    assert (snapshot / 'COMPLETE.json').is_file()
    backup.verify(snapshot)
    backup.restore(snapshot, destination, 64)
    assert (destination / 'asset').read_bytes() == b'payload'
    backup.publish(source, snapshot, 'assets', 64)
    (source / 'asset').write_bytes(b'changed')
    with pytest.raises(ValueError, match='immutable'):
        backup.publish(source, snapshot, 'assets', 64)


def test_remote_corruption_and_marker_tampering(tmp_path):
    source, snapshot = tmp_path / 'source', tmp_path / 'snapshot'
    source.mkdir()
    (source / 'asset').write_bytes(b'payload')
    backup.publish(source, snapshot, 'assets', 64)
    (snapshot / 'payload/asset').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        backup.verify(snapshot)
    (snapshot / 'FILES.json').write_text('{}')
    with pytest.raises(ValueError, match='disagree'):
        backup.verify(snapshot)


def test_incomplete_payload_is_repaired(tmp_path):
    source, snapshot = tmp_path / 'source', tmp_path / 'snapshot'
    source.mkdir()
    (source / 'asset').write_bytes(b'payload')
    (snapshot / 'payload').mkdir(parents=True)
    (snapshot / 'payload/asset').write_bytes(b'part')
    backup.publish(source, snapshot, 'assets', 64)
    assert backup.verify(snapshot)['files'][0]['bytes'] == 7


@pytest.mark.parametrize('name', ['.env.local', 'secret.key', 'weights.part'])
def test_private_or_partial_files_rejected(tmp_path, name):
    (tmp_path / name).write_text('secret')
    with pytest.raises(ValueError):
        backup.source_files(tmp_path, 'assets')


def test_symlink_and_wrong_namespace_rejected(tmp_path):
    (tmp_path / 'link').symlink_to('/tmp', target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        backup.source_files(tmp_path, 'assets')
    with pytest.raises(ValueError, match='dedicated namespace'):
        backup.check_blob_root(Path('/mnt/blob-data-sigmasystem-out/juexiao/spmem-runtime'))


def test_restore_does_not_overwrite_existing(tmp_path):
    source, snapshot, destination = tmp_path / 'source', tmp_path / 'snapshot', tmp_path / 'restore'
    source.mkdir()
    destination.mkdir()
    (source / 'asset').write_bytes(b'payload')
    (destination / 'asset').write_bytes(b'user data')
    backup.publish(source, snapshot, 'assets', 64)
    with pytest.raises(ValueError, match='refuses to overwrite'):
        backup.restore(snapshot, destination, 64)
    assert (destination / 'asset').read_bytes() == b'user data'


def test_completion_marker_binds_manifest(tmp_path):
    source, snapshot = tmp_path / 'source', tmp_path / 'snapshot'
    source.mkdir()
    (source / 'asset').write_text('data')
    backup.publish(source, snapshot, 'assets', 64)
    marker = json.loads((snapshot / 'COMPLETE.json').read_text())
    assert marker['manifest_sha256'] == backup.digest(snapshot / 'FILES.json')
