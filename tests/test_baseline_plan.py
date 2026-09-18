from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_script('prepare_nav_assets')
load_script('preflight_nav_assets')
planner = load_script('plan_nav_baseline')


def test_plan_is_nonlaunching_and_preserves_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(planner, 'audit', lambda *args: {'errors': [], 'local_asset_checks_passed': True})
    plan = planner.build_plan(tmp_path, tmp_path / 'scenes', tmp_path / 'plan', [2, 5],
                              Path(sys.executable), Path(sys.executable), 'hf', -1, 5555)
    assert plan['launch_ready'] is False
    assert plan['processes_started'] == 0
    assert plan['requires_explicit_gpu_authorization']
    assert not plan['errors']
    rxr = plan['benchmarks']['rxr']['environment']
    assert rxr['GPU_IDS'] == '2 5'
    assert rxr['LANGUAGES'] == 'en-US en-IN'
    assert 'DATA_PATH' not in rxr
    config = yaml.safe_load(Path(rxr['HABITAT_CONFIG']).read_text())
    assert config['habitat']['dataset']['data_path'].endswith('{split}/{split}_guide.json.gz')
    assert config['habitat']['simulator']['agents']['main_agent']['sim_sensors']['rgb_sensor']['hfov'] == 120
    assert json.loads((tmp_path / 'plan/plan.json').read_text())['method'] == 'B0_original_slowfast'


@pytest.mark.parametrize('ids', [[], [0, 0], [-1]])
def test_gpu_ids_must_be_explicit_and_valid(tmp_path, ids):
    with pytest.raises(ValueError, match='GPU IDs'):
        planner.build_plan(tmp_path, tmp_path, tmp_path / 'plan', ids,
                           Path(sys.executable), Path(sys.executable), 'hf', -1, 5555)


def test_missing_assets_stay_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(planner, 'audit', lambda *args: {'errors': ['Missing MP3D'], 'local_asset_checks_passed': False})
    plan = planner.build_plan(tmp_path, tmp_path, tmp_path / 'plan', [0],
                              Path(sys.executable), tmp_path / 'missing-python', 'hf', 1, 5555)
    assert 'Missing MP3D' in plan['errors']
    assert any('Missing interpreter' in error for error in plan['errors'])
    assert plan['processes_started'] == 0


def test_existing_plan_not_overwritten(tmp_path):
    (tmp_path / 'plan.json').write_text('user plan')
    with pytest.raises(ValueError, match='already exists'):
        planner.build_plan(tmp_path, tmp_path, tmp_path, [0],
                           Path(sys.executable), Path(sys.executable), 'hf', -1, 5555)


def test_cli_preserves_venv_interpreter_symlink(tmp_path, monkeypatch):
    interpreter = tmp_path / 'venv/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    captured = {}

    def fake_plan(*args):
        captured['interpreter'] = args[4]
        return {'errors': []}

    monkeypatch.setattr(planner, 'build_plan', fake_plan)
    monkeypatch.setattr(sys, 'argv', ['plan_nav_baseline.py', '--data-root', str(tmp_path),
                                    '--output-root', str(tmp_path / 'plan'), '--gpu-ids', '0',
                                    '--model-python', str(interpreter)])
    assert planner.main() == 0
    assert captured['interpreter'] == interpreter.absolute()
    assert captured['interpreter'] != interpreter.resolve()
