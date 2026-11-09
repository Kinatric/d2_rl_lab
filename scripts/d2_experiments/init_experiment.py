"""Create a private experiment: python scripts/d2_experiments/init_experiment.py --output tests/upgrade_isaaclab_3_ea/my_run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from _paths import HERE, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=HERE / 'experiment.example.json')
    parser.add_argument('--baseline-policy', type=Path, help='Optional exported D2 TorchScript baseline; keep its config.yaml alongside it.')
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    private_root = (ROOT / 'tests').resolve()
    if not out.is_relative_to(private_root) or out == private_root:
        parser.error('--output must be a new subdirectory of this checkout\'s tests directory')
    if out.exists():
        parser.error('Output already exists; choose a new experiment directory')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    fixed = {'task': 'd2-mixed-moe-cts', 'num_envs': 8192, 'seed': 42,
             'physics': 'newton_mjwarp', 'zero_command_end': .1,
             'zero_yaw_probability': 0, 'teacher_envs': 6144,
             'student_envs': 2048, 'save_interval': 500}
    if any(config.get(k) != v for k, v in fixed.items()):
        parser.error('This workflow supports the fixed MoE configuration in experiment.example.json')
    target = config.get('target_iterations')
    if type(target) is not int or target < 501:
        parser.error('target_iterations must be an integer >= 501')
    if args.baseline_policy:
        baseline = args.baseline_policy.expanduser().resolve()
        if not baseline.is_file() or not (baseline.parent / 'config.yaml').is_file():
            parser.error('Baseline requires policy.pt and its adjacent config.yaml')
        config['baseline_policy'] = str(baseline)
    else:
        config['baseline_policy'] = None
    # Resolve Git inventory before making any output, including a useful error for non-clones.
    tracked = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT
    ).decode().split('\0')
    out.mkdir(parents=True)
    for source in HERE.glob('*.py'):
        if source.name != 'init_experiment.py':
            shutil.copy2(source, out / source.name)
    (out / 'experiment.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    paths = [ROOT / name for name in tracked if name and not name.startswith('tests/')]
    paths += list(HERE.glob('*.py')) + list(out.glob('*.py')) + [out / 'experiment.json']
    manifest = {'upstream_commit': '6c95d9feef4814c25eb6536e61bb38e1d1d718d4',
                'files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths if p.is_file()}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Created {out}\nRun preflight.py, then evaluation_smoke.py, then workflow.py inside that directory.')


if __name__ == '__main__':
    main()
