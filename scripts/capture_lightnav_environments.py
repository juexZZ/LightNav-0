"""Capture sanitized dependency locks and CPU-import receipts without model loading."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from prepare_nav_assets import write_json

REPO = Path(__file__).resolve().parents[1]


def capture(output: Path, model_python: Path, habitat_python: Path, micromamba: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty receipt directory')
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        'PATH': os.environ['PATH'], 'HOME': str(Path.home()), 'LANG': 'C.UTF-8',
        'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'PYTHONNOUSERSITE': '1',
        'MAMBA_ROOT_PREFIX': str(Path.home() / 'lightnav_tools/mamba'),
    }

    def run(name: str, argv: list[str]) -> str:
        result = subprocess.run(argv, cwd=REPO, env=environment, capture_output=True, text=True)
        (output / f'{name}.stderr.txt').write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f'{name} failed: see {output / (name + ".stderr.txt")}')
        text = result.stdout.replace(str(REPO / 'habitat_server'), './habitat_server').replace(str(REPO), '.')
        (output / f'{name}.txt').write_text(text)
        return result.stdout

    model_check = run('model-import', [str(model_python), '-c',
        'import json, sys, torch, importlib.metadata as metadata; import lightnav.cli.eval_habitat; '
        'assert not torch.cuda.is_initialized(); '
        'print(json.dumps({"python":sys.version.split()[0],"torch":torch.__version__, '
        '"transformers":metadata.version("transformers"),"vllm":metadata.version("vllm"),'
        '"cutlass":metadata.version("nvidia-cutlass-dsl"),"cuda_initialized":False}))'])
    habitat_check = run('habitat-import', [str(habitat_python), '-c',
        'import json, sys, habitat, habitat_sim, lightnav_habitat.serve, numpy; '
        'print(json.dumps({"python":sys.version.split()[0],"numpy":numpy.__version__, '
        '"habitat_sim":getattr(habitat_sim,"__version__",None),"simulator_created":False}))'])
    run('model-pip-freeze', ['uv', 'pip', 'freeze', '--python', str(model_python)])
    run('habitat-pip-freeze', ['uv', 'pip', 'freeze', '--python', str(habitat_python)])
    run('habitat-conda-explicit', [str(micromamba), 'list', '-p', str(habitat_python.parent.parent), '--explicit'])
    summary = {
        'scope': 'CPU imports and dependency locks only', 'gpu_verified': False,
        'model': json.loads(model_check.strip().splitlines()[-1]),
        'habitat': json.loads(habitat_check.strip().splitlines()[-1]),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'git_worktree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()),
    }
    write_json(output / 'ENVIRONMENT_RECEIPT.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-python', type=Path, default=REPO / '.venv/bin/python')
    parser.add_argument('--habitat-python', type=Path, default=Path.home() / 'envs/lightnav-habitat/bin/python')
    parser.add_argument('--micromamba', type=Path, default=Path.home() / 'lightnav_tools/micromamba')
    args = parser.parse_args()
    print(json.dumps(capture(args.output, args.model_python.absolute(), args.habitat_python.absolute(), args.micromamba), indent=2))


if __name__ == '__main__':
    main()
