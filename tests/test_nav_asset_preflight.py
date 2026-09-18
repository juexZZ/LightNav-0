from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_nav_assets")
preflight = load_script("preflight_nav_assets")
MANIFEST = json.loads((ROOT / "configs/research/nav_assets.json").read_text())


def write_gzip(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as stream:
        json.dump(payload, stream)


def episode(identifier="1", language=None, scene="mp3d/example/example.glb"):
    return {
        "episode_id": identifier, "scene_id": scene,
        "instruction": {"instruction_text": "Walk forward.", "language": language},
        "start_position": [0, 0, 0], "start_rotation": [0, 0, 0, 1],
        "reference_path": [[0, 0, 0], [0, 0, 1]],
    }


def make_dataset(root, spec, records, split="val_unseen", gt=True, scene=True):
    directory = root / "data/datasets" / spec["directory"] / split
    write_gzip(directory / spec["episodes"].format(split=split), {"episodes": records})
    if gt:
        ground_truth = {str(record["episode_id"]): {"locations": [[0, 0, 0], [0, 0, 1]]} for record in records}
        write_gzip(directory / spec["ground_truth"].format(split=split), ground_truth)
    if scene:
        path = root / "data/scene_datasets/mp3d/example/example.glb"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12))


def make_model(root):
    model_root = root / "models/LightNav-0"
    for name in MANIFEST["model"]["files"]:
        path = model_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}" if path.suffix == ".json" else b"fixture")
    prepare.write_json(model_root / "eval_config.json", {"tasks": {"vlnce": {"num_history_frames": 64}}})
    prepare.write_json(model_root / "model.safetensors.index.json", {"weight_map": {"weight": "model-00001-of-00001.safetensors"}})
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (model_root / "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    prepare.write_json(model_root / "action_tokenizer/manifest.json", {
        "horizon": 10, "levels": [256, 256, 256], "codebook_files": ["codebook_l0.npy"],
        "jacobian_weights_file": "jacobian_weights.npy", "alpha_file": "alpha_per_source.json",
    })
    provenance = {
        "repo_id": MANIFEST["model"]["repo_id"], "revision": MANIFEST["model"]["revision"],
        "files": {name: prepare.hashes(model_root / name) for name in MANIFEST["model"]["files"]},
    }
    prepare.write_json(model_root / "asset_provenance.json", provenance)
    return model_root


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../../escape", "a\\escape"])
def test_safe_path_rejects_traversal(tmp_path, name):
    with pytest.raises(ValueError):
        prepare.safe_path(tmp_path, name)


def test_safe_path_rejects_symlink_escape(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        prepare.safe_path(tmp_path, "link/escape")


def test_publisher_checksums(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"correct")
    actual = prepare.hashes(path)
    assert prepare.verify_file(path, {"size": 7, "lfs": {"sha256": actual["sha256"]}}) == actual
    assert prepare.verify_file(path, {"size": 7, "blobId": actual["git_blob_sha1"]}) == actual
    with pytest.raises(ValueError, match="checksum"):
        prepare.verify_file(path, {"size": 7, "blobId": "wrong"})


def test_extract_only_selected_annotations(tmp_path):
    archive = tmp_path / "dataset.zip"
    spec = MANIFEST["datasets"]["r2r"]
    name = f"{spec['directory']}/val_unseen/val_unseen.json.gz"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(name, gzip.compress(b'{"episodes": []}'))
        bundle.writestr(f"{spec['directory']}/text_features/unused.npy", b"unused")
    result = prepare.extract_annotations(archive, tmp_path / "data", spec, ["val_unseen"])
    assert [path.relative_to(tmp_path / "data").as_posix() for path in result] == [name]
    assert not (tmp_path / "data" / spec["directory"] / "text_features").exists()


def test_extract_rejects_zip_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.json.gz", b"bad")
    with pytest.raises(ValueError, match="Unsafe"):
        prepare.extract_annotations(archive, tmp_path / "data", MANIFEST["datasets"]["r2r"], ["val_unseen"])


def test_complete_pinned_model_checks_without_loading(tmp_path):
    model_root = make_model(tmp_path)
    assert not preflight.check_model(model_root, MANIFEST["model"], True)["errors"]


def test_modified_model_checksum_is_rejected(tmp_path):
    model_root = make_model(tmp_path)
    (model_root / "README.md").write_bytes(b"changed")
    errors = preflight.check_model(model_root, MANIFEST["model"], True)["errors"]
    assert any("SHA-256" in error for error in errors)


def test_partial_model_does_not_pass(tmp_path):
    model_root = make_model(tmp_path)
    shard = model_root / "model-00001-of-00001.safetensors"
    shard.rename(shard.with_suffix(".part"))
    assert preflight.check_model(model_root, MANIFEST["model"])["errors"]


def test_truncated_safetensors_is_rejected(tmp_path):
    model_root = make_model(tmp_path)
    shard = model_root / "model-00001-of-00001.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        preflight.check_safetensors(shard)


def test_rxr_guide_language_filter_and_gt(tmp_path):
    spec = {**MANIFEST["datasets"]["rxr"], "val_unseen_count": 2}
    make_dataset(tmp_path, spec, [episode("1", "en-US"), episode("2", "en-IN"), episode("3", "hi-IN")])
    result = preflight.check_dataset(tmp_path / "data/datasets", tmp_path / "data/scene_datasets", spec, "val_unseen")
    assert result["raw_count"] == 3
    assert result["selected_count"] == 2
    assert result["ground_truth_path"].endswith("val_unseen_guide_gt.json.gz")
    assert not result["errors"]


def test_missing_gt_blocks_readiness(tmp_path):
    spec = {**MANIFEST["datasets"]["r2r"], "val_unseen_count": 1}
    make_dataset(tmp_path, spec, [episode()], gt=False)
    result = preflight.check_dataset(tmp_path / "data/datasets", tmp_path / "data/scene_datasets", spec, "val_unseen")
    assert any("_gt.json.gz" in error for error in result["errors"])


def test_duplicates_and_missing_mp3d_are_reported(tmp_path):
    spec = {**MANIFEST["datasets"]["r2r"], "val_unseen_count": 2}
    make_dataset(tmp_path, spec, [episode(), episode()], scene=False)
    result = preflight.check_dataset(tmp_path / "data/datasets", tmp_path / "data/scene_datasets", spec, "val_unseen")
    assert len(result["missing_scenes"]) == 1
    assert any("Duplicate" in error for error in result["errors"])


def test_hm3d_not_accepted_as_mp3d(tmp_path):
    (tmp_path / "hm3d/val").mkdir(parents=True)
    result = preflight.audit(MANIFEST, tmp_path, tmp_path, ["val_unseen"], False)
    assert result["detected_scene_families"] == ["hm3d"]
    assert not result["local_asset_checks_passed"]
    assert not result["gpu_verified"]
    assert any("HM3D is not" in error for error in result["errors"])


def test_hm3d_inventory_scene_ids_are_compared(tmp_path):
    write_gzip(tmp_path / "versioned_data/hm3d-0.2/train-glb-files.json.gz", [
        "/hm3d/train/00000-hmexample", "/hm3d/train/00000-hmexample/hmexample.glb",
    ])
    result = preflight.hm3d_inventory_comparison(tmp_path, {"mp3d/example/example.glb"})
    assert result["hm3d_scene_ids"] == 1
    assert result["required_mp3d_scene_ids"] == 1
    assert result["matching_ids"] == []


def test_split_scene_leakage_is_reported(tmp_path):
    manifest = copy.deepcopy(MANIFEST)
    manifest["datasets"] = {"r2r": {**MANIFEST["datasets"]["r2r"], "val_unseen_count": 1}}
    spec = manifest["datasets"]["r2r"]
    make_model(tmp_path)
    make_dataset(tmp_path, spec, [episode()], split="train")
    make_dataset(tmp_path, spec, [episode()], split="val_unseen")
    result = preflight.audit(manifest, tmp_path, tmp_path / "data/scene_datasets", ["train", "val_unseen"], False)
    assert result["train_val_unseen_scene_overlap"] == ["mp3d/example/example.glb"]
    assert not result["local_asset_checks_passed"]


def test_cli_writes_blocked_report_without_optional_runtime(tmp_path):
    output = tmp_path / "audit.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/preflight_nav_assets.py"),
         "--data-root", str(tmp_path), "--output", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "BLOCKED" in result.stdout
    assert json.loads(output.read_text())["gpu_verified"] is False
