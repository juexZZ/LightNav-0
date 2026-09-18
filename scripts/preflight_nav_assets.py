"""Read-only, CPU-only checks for LightNav weights and R2R/RxR VLN-CE assets."""

from __future__ import annotations

import argparse
import gzip
import json
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from prepare_nav_assets import DEFAULT_MANIFEST, hashes, safe_path, write_json


def read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def check_safetensors(path: Path) -> None:
    with path.open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < header_length <= 32 * 1024**2:
            raise ValueError(f"Invalid safetensors header length: {path.name}")
        header = json.loads(stream.read(header_length))
    data_size = path.stat().st_size - 8 - header_length
    intervals = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        start, end = tensor["data_offsets"]
        if not 0 <= start <= end <= data_size:
            raise ValueError(f"Truncated tensor data: {path.name}/{name}")
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError(f"Noncontiguous tensor data: {path.name}")
        cursor = end
    if not intervals or cursor != data_size:
        raise ValueError(f"Unexpected safetensors payload size: {path.name}")


def check_model(model_root: Path, spec: dict, verify_sha256: bool = False) -> dict:
    report = {"path": str(model_root), "errors": [], "sha256_rechecked": verify_sha256}
    errors = report["errors"]
    for name in spec["files"]:
        path = safe_path(model_root, name)
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty model asset: {name}")
    provenance_path = model_root / "asset_provenance.json"
    if not provenance_path.is_file():
        errors.append("Missing completed publisher-checksum verification: asset_provenance.json")
    if errors:
        return report
    provenance = read_json(provenance_path)
    if (provenance.get("repo_id"), provenance.get("revision")) != (spec["repo_id"], spec["revision"]):
        errors.append("Checkpoint provenance does not match the pinned model revision")
    for name in spec["files"]:
        path = safe_path(model_root, name)
        recorded = provenance.get("files", {}).get(name, {})
        if recorded.get("size_bytes") != path.stat().st_size or not recorded.get("sha256"):
            errors.append(f"Missing or inconsistent verified file metadata: {name}")
        elif verify_sha256 and hashes(path)["sha256"] != recorded["sha256"]:
            errors.append(f"SHA-256 mismatch: {name}")
        if name.endswith(".json"):
            read_json(path)
        if name.endswith(".safetensors"):
            check_safetensors(path)
    index = read_json(model_root / "model.safetensors.index.json")
    shards = set(index.get("weight_map", {}).values())
    if not shards:
        errors.append("Empty safetensors weight map")
    for shard in shards:
        if shard not in spec["files"] or not safe_path(model_root, shard).is_file():
            errors.append(f"Missing indexed model shard: {shard}")
    bundle = read_json(model_root / "action_tokenizer/manifest.json")
    required = [*bundle.get("codebook_files", []), *bundle.get("distance_files", [])]
    required += [bundle["jacobian_weights_file"], bundle["alpha_file"]]
    for name in required:
        if not safe_path(model_root / "action_tokenizer", name).is_file():
            errors.append(f"Missing RVQ dependency: {name}")
    if bundle.get("horizon") != 10 or bundle.get("levels") != [256, 256, 256]:
        errors.append("RVQ configuration differs from the pinned release")
    report["revision"] = provenance.get("revision")
    report["vlnce_config"] = read_json(model_root / "eval_config.json").get("tasks", {}).get("vlnce")
    return report


def mp3d_relative_path(scene_id: str) -> str:
    prefix = "data/scene_datasets/"
    if scene_id.startswith(prefix):
        scene_id = scene_id[len(prefix):]
    parts = PurePosixPath(scene_id).parts
    if len(parts) != 3 or parts[0] != "mp3d" or parts[1] in {".", ".."}:
        raise ValueError(f"Not a standard MP3D scene path: {scene_id}")
    if parts[2] != parts[1] + ".glb" or "\\" in scene_id:
        raise ValueError(f"Invalid MP3D scene filename: {scene_id}")
    return str(PurePosixPath(*parts))


def scene_families(root: Path) -> list[str]:
    families = []
    if (root / "mp3d").is_dir():
        families.append("mp3d")
    candidates = [root / "hm3d", root / "versioned_data/hm3d-0.2/hm3d"]
    if root.name == "hm3d" or any(path.is_dir() for path in candidates):
        families.append("hm3d")
    return families


def hm3d_inventory_comparison(root: Path, required_scenes: set[str]) -> dict | None:
    inventory_root = root / "versioned_data/hm3d-0.2"
    manifests = sorted(inventory_root.glob("*-glb-files.json.gz"))
    if not manifests:
        return None
    available = set()
    for path in manifests:
        for entry in read_json(path):
            name = PurePosixPath(entry)
            if name.suffix == ".glb" and not name.name.endswith(".basis.glb"):
                available.add(name.stem)
    required = {PurePosixPath(scene).stem for scene in required_scenes}
    return {
        "scope": "Inventory names only; not a full HM3D file-integrity audit",
        "hm3d_scene_ids": len(available), "required_mp3d_scene_ids": len(required),
        "matching_ids": sorted(available & required),
    }


def check_dataset(dataset_root: Path, scenes_root: Path, spec: dict, split: str) -> dict:
    report = {"split": split, "errors": [], "scenes": [], "missing_scenes": []}
    errors = report["errors"]
    split_root = dataset_root / spec["directory"] / split
    episodes_path = split_root / spec["episodes"].format(split=split)
    gt_path = split_root / spec["ground_truth"].format(split=split)
    report["episodes_path"], report["ground_truth_path"] = str(episodes_path), str(gt_path)
    for path in (episodes_path, gt_path):
        if not path.is_file():
            errors.append(f"Missing annotation file: {path}")
    if errors:
        return report
    payload, ground_truth = read_json(episodes_path), read_json(gt_path)
    episodes = payload["episodes"]
    selected, languages = [], Counter()
    for episode in episodes:
        instruction = episode.get("instruction", {})
        language = instruction.get("language") if isinstance(instruction, dict) else None
        languages[str(language)] += 1
        if not spec["languages"] or language in spec["languages"]:
            selected.append(episode)
    report.update(raw_count=len(episodes), selected_count=len(selected), languages=dict(languages))
    if not selected:
        errors.append("No episodes selected after language filtering")
    if split == "val_unseen" and len(selected) != spec["val_unseen_count"]:
        errors.append(f"Expected {spec['val_unseen_count']} selected val_unseen episodes, got {len(selected)}")
    seen_keys, scenes = set(), set()
    for episode in selected:
        identifier = str(episode["episode_id"])
        scene = mp3d_relative_path(episode["scene_id"])
        key = (scene, identifier)
        if key in seen_keys:
            errors.append(f"Duplicate episode key: {key}")
        seen_keys.add(key)
        scenes.add(scene)
        instruction = episode.get("instruction", {})
        text = instruction.get("instruction_text", instruction.get("text", "")) if isinstance(instruction, dict) else instruction
        if not isinstance(text, str) or not text.strip():
            errors.append(f"Missing instruction text: {identifier}")
        if len(episode.get("start_position", [])) != 3 or len(episode.get("start_rotation", [])) != 4:
            errors.append(f"Invalid start pose: {identifier}")
        if not episode.get("reference_path"):
            errors.append(f"Missing reference_path: {identifier}")
        if not isinstance(ground_truth.get(identifier), dict) or not ground_truth[identifier].get("locations"):
            errors.append(f"Missing nDTW ground-truth locations: {identifier}")
    report["scenes"] = sorted(scenes)
    report["scene_count"] = len(scenes)
    for scene in sorted(scenes):
        path = safe_path(scenes_root, scene)
        if not path.is_file():
            report["missing_scenes"].append(scene)
            continue
        with path.open("rb") as stream:
            header = stream.read(12)
        if len(header) != 12:
            errors.append(f"Truncated GLB header: {scene}")
        else:
            magic, version, length = struct.unpack("<4sII", header)
            if magic != b"glTF" or version != 2 or length != path.stat().st_size:
                errors.append(f"Invalid or truncated GLB: {scene}")
    if report["missing_scenes"]:
        errors.append(f"Missing {len(report['missing_scenes'])} required MP3D scene files")
    report["annotations_sha256"] = {path.name: hashes(path)["sha256"] for path in (episodes_path, gt_path)}
    return report


def audit(manifest: dict, root: Path, scenes_root: Path, splits: list[str], verify_sha256: bool) -> dict:
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU file/schema checks only; no model load, CUDA, Habitat, or rendering",
        "gpu_verified": False, "scenes_root": str(scenes_root),
        "detected_scene_families": scene_families(scenes_root), "datasets": {}, "errors": [],
    }
    try:
        report["model"] = check_model(root / "models/LightNav-0", manifest["model"], verify_sha256)
    except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
        report["model"] = {"errors": [str(error)]}
    report["errors"].extend(f"model: {error}" for error in report["model"]["errors"])
    training_scenes, unseen_scenes, required_scenes = set(), set(), set()
    for name, spec in manifest["datasets"].items():
        for split in splits:
            key = f"{name}/{split}"
            try:
                result = check_dataset(root / "data/datasets", scenes_root, spec, split)
            except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
                result = {"errors": [str(error)], "scenes": []}
            report["datasets"][key] = result
            report["errors"].extend(f"{key}: {error}" for error in result["errors"])
            required_scenes.update(result["scenes"])
            if split == "train":
                training_scenes.update(result["scenes"])
            elif split == "val_unseen":
                unseen_scenes.update(result["scenes"])
    report["train_val_unseen_scene_overlap"] = sorted(training_scenes & unseen_scenes)
    if report["train_val_unseen_scene_overlap"]:
        report["errors"].append("Training scenes overlap held-out val_unseen scenes")
    if "hm3d" in report["detected_scene_families"] and "mp3d" not in report["detected_scene_families"]:
        report["errors"].append("HM3D is not the MP3D scene set required by R2R/RxR; do not rename or substitute it")
        try:
            report["hm3d_inventory_comparison"] = hm3d_inventory_comparison(scenes_root, required_scenes)
        except (OSError, ValueError, TypeError, EOFError) as error:
            report["errors"].append(f"Cannot read HM3D inventory: {error}")
    report["required_mp3d_scenes"] = sorted(required_scenes)
    report["local_asset_checks_passed"] = not report["errors"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scenes-dir", type=Path)
    parser.add_argument("--splits", nargs="+", choices=["train", "val_seen", "val_unseen"], default=["val_unseen"])
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    scenes_root = args.scenes_dir or args.data_root / "data/scene_datasets"
    report = audit(manifest, args.data_root, scenes_root, args.splits, args.verify_sha256)
    write_json(args.output, report)
    state = "PASS" if report["local_asset_checks_passed"] else "BLOCKED"
    print(f"{state}: {len(report['errors'])} issue(s); report={args.output}; GPU verified=False")
    for error in report["errors"][:12]:
        print(f"  {error}")
    return 0 if report["local_asset_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
