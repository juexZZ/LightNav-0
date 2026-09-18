"""Generate B0 configs and command plans only; never launch a GPU or Habitat process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from preflight_nav_assets import audit
from prepare_nav_assets import DEFAULT_MANIFEST, write_json

REPO = Path(__file__).resolve().parents[1]


def build_plan(data_root: Path, scenes_root: Path, output: Path, gpu_ids: list[int],
               model_python: Path, habitat_python: Path, backend: str, episodes: int,
               base_port: int) -> dict:
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or min(gpu_ids) < 0:
        raise ValueError('Explicit, unique, non-negative GPU IDs are required')
    if episodes != -1 and episodes < 1:
        raise ValueError('episodes must be -1 or a positive per-shard count')
    if not 1024 <= base_port <= 65535 - len(gpu_ids) + 1:
        raise ValueError('Invalid port range')
    if (output / 'plan.json').exists():
        raise ValueError('Plan already exists; use a new output directory')
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    assets = audit(manifest, data_root, scenes_root, ['val_unseen'], False)
    errors = list(assets['errors'])
    for interpreter in (model_python, habitat_python):
        if not interpreter.is_file():
            errors.append(f'Missing interpreter: {interpreter}')
    plan = {
        'mode': 'dry_run_only', 'method': 'B0_original_slowfast', 'launch_ready': False,
        'requires_explicit_gpu_authorization': True, 'processes_started': 0,
        'errors': errors, 'asset_checks_passed': assets['local_asset_checks_passed'],
        'model_revision': manifest['model']['revision'], 'benchmarks': {},
        'execution_order': ['r2r', 'rxr'],
        'notes': ['Run sequentially; never auto-launch on idle GPUs.',
                  'episodes is per shard; -1 evaluates the full split.',
                  'Rendered frames, decoding, controller and STOP follow the original evaluator.'],
    }
    for name, dataset in manifest['datasets'].items():
        source = REPO / 'habitat_server/configs' / f'vlnce_{name}.yaml'
        original = source.read_text()
        config = yaml.safe_load(original)
        config['habitat']['dataset']['data_path'] = str(
            data_root / 'data/datasets' / dataset['directory'] / '{split}'
            / dataset['episodes'].format(split='{split}')
        )
        config['habitat']['dataset']['scenes_dir'] = str(scenes_root)
        target = output / 'configs' / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(original.splitlines()[0] + '\n' + yaml.safe_dump(config, sort_keys=False))
        environment = {
            'MODEL_PATH': str(data_root / 'models/LightNav-0'), 'BACKEND': backend,
            'HABITAT_CONFIG': str(target), 'SCENES_DIR': str(scenes_root),
            'SPLIT': 'val_unseen', 'TASK': 'vlnce', 'MAX_STEPS': '500',
            'GPU_IDS': ' '.join(map(str, gpu_ids)), 'BASE_PORT': str(base_port),
            'CLIENT_PYTHON': str(model_python), 'HABITAT_PYTHON': str(habitat_python),
            'OUTPUT_ROOT': str(output / name), 'EPISODES': str(episodes),
        }
        if dataset['languages']:
            environment['LANGUAGES'] = ' '.join(dataset['languages'])
        plan['benchmarks'][name] = {
            'environment': environment, 'argv': ['bash', str(REPO / 'scripts/eval_habitat.sh')],
            'cwd': str(REPO), 'expected_full_split_episodes': dataset['val_unseen_count'],
            'source_config_sha256': hashlib.sha256(original.encode()).hexdigest(),
        }
    write_json(output / 'asset_preflight.json', assets)
    write_json(output / 'plan.json', plan)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--scenes-dir', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpu-ids', nargs='+', type=int, required=True)
    parser.add_argument('--model-python', type=Path, default=REPO / '.venv/bin/python')
    parser.add_argument('--habitat-python', type=Path, default=Path.home() / 'envs/lightnav-habitat/bin/python')
    parser.add_argument('--backend', choices=['hf', 'vllm_local'], default='vllm_local')
    parser.add_argument('--episodes', type=int, default=-1)
    parser.add_argument('--base-port', type=int, default=5555)
    args = parser.parse_args()
    scenes = args.scenes_dir or args.data_root / 'data/scene_datasets'
    plan = build_plan(args.data_root.resolve(), scenes.resolve(), args.output_root.resolve(),
                      args.gpu_ids, args.model_python.absolute(), args.habitat_python.absolute(),
                      args.backend, args.episodes, args.base_port)
    print(f'DRY_RUN: processes_started=0; launch_ready=False; issues={len(plan["errors"])}')
    print(f'Plan: {args.output_root / "plan.json"}')
    for error in plan['errors']:
        print(error)
    return 2 if plan['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
