from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_script('prepare_nav_assets')
capture = load_script('capture_lightnav_environments')


def test_receipt_filters_environment_and_normalizes_editable_path(tmp_path, monkeypatch):
    monkeypatch.setenv('GITHUB_TOKEN', 'must_not_be_copied')

    def fake_run(argv, **kwargs):
        assert 'GITHUB_TOKEN' not in kwargs['env']
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        if argv[0] == 'uv':
            output = f'-e {capture.REPO}\nexample==1.0\n'
        elif 'torch' in argv[-1]:
            output = json.dumps({'cuda_initialized': False})
        elif 'habitat_sim' in argv[-1]:
            output = json.dumps({'simulator_created': False})
        else:
            output = '@EXPLICIT\nhttps://conda.anaconda.org/example.conda\n'
        return SimpleNamespace(returncode=0, stdout=output, stderr='')

    monkeypatch.setattr(capture.subprocess, 'run', fake_run)
    monkeypatch.setattr(capture.subprocess, 'check_output', lambda argv, **kwargs: 'commit\n' if 'rev-parse' in argv else '')
    output = tmp_path / 'receipt'
    result = capture.capture(output, Path('/model/python'), Path('/habitat/bin/python'), Path('/micromamba'))
    assert result['gpu_verified'] is False
    assert result['git_worktree_dirty'] is False
    assert (output / 'model-pip-freeze.txt').read_text().startswith('-e .\n')
    assert all('must_not_be_copied' not in path.read_text() for path in output.iterdir())


def test_capture_refuses_existing_receipts(tmp_path):
    (tmp_path / 'existing').write_text('keep')
    with pytest.raises(ValueError, match='empty'):
        capture.capture(tmp_path, Path('/python'), Path('/python'), Path('/micromamba'))
