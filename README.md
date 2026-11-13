# D2_rl_lab

D2 locomotion with Isaac Lab 3.0, Newton/PhysX and MuJoCo student-policy deployment. Based on [wty-yy/go2_rl_lab](https://github.com/wty-yy/go2_rl_lab), `lab3.0` commit `6c95d9feef4814c25eb6536e61bb38e1d1d718d4`. Go2 robot support is removed from the current checkout; upstream license, attribution and Git history remain. No trained policies or generated USD assets are distributed.

## Packages and tasks

- `source/robot_rl_core`: shared velocity commands, MDP functions, terrains, USD loading, Newton physics and utilities.
- `source/d2_rl_lab`: D2 description, RS06 actuator, physical/observation identity, symmetry, environment and network configuration, deployment adapter.
- `source/rsl_rl`: the sole project-specific PPO/MoE-CTS implementation.

| Task | Purpose |
| --- | --- |
| `d2-flat` | Flat-ground PPO |
| `d2-mixed` | Mixed-terrain PPO |
| `d2-mixed-moe-cts` | Mixed-terrain concurrent teacher/student training, 8 student experts |

D2 uses 12 joints, a 45-value frame with 10-frame history, 200 Hz physics and 50 Hz policy control. Those robot-specific settings belong to D2, not the common core. D2W is not yet implemented.

## Install and generate assets

Current source assets are stored directly in Git. Cloning or downloading the latest ZIP does not require Git LFS. Older commits or tags may still require Git LFS; download a fresh archive to replace an older archive containing pointer files.

Use Python 3.12 with the configured Isaac Lab 3.0 / Isaac Sim 6.2 runtime (PyTorch 2.11 CUDA 12.8, Newton 1.6.1, Warp 1.17). See the [upstream runtime installation](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/installation.html). Use a separate environment when migrating; do not change active jobs' editable paths.

```bash
git clone http://192.168.0.100:3002/lixiang/D2_rl_lab
cd D2_rl_lab
# Activate your independent Isaac Lab environment first.
./d2_rl_lab.sh --install
./d2_rl_lab.sh --list
```

The install entry installs local RSL-RL, core, then D2 with `--no-deps`; it does not install/upgrade the simulation stack. Optional video/MuJoCo extras use the existing dependency helper. Keep the local RSL-RL 3.3.0 implementation despite Isaac Lab's newer upstream dependency declaration.

Before training, follow [asset generation and three-backend standing validation](docs/d2_assets.md). Original URDF/meshes/configs live in `source/d2_rl_lab/d2_rl_lab/assets/d2_description/`; generated assets stay in ignored `resources/d2_runtime/`. Existing verified assets can be copied without conversion; see [migration](docs/core_migration.md). D2 task selection validates assets; task listing does not require them.

## Training and recovery

```bash
./d2_rl_lab.sh --train --task d2-flat --physics newton_mjwarp \
  --viz none --num_envs 32 --max_iterations 2 --logger tensorboard

./d2_rl_lab.sh --train --task d2-mixed-moe-cts --physics isaacsim_physx \
  --viz none --num_envs 32 --max_iterations 2 --logger tensorboard

./d2_rl_lab.sh --train --task d2-mixed-moe-cts --physics newton_mjwarp \
  --viz none --num_envs 32 --resume --checkpoint /path/to/model.pt \
  --max_iterations 1 --logger tensorboard
```

`--max_iterations` in the plain CLI counts iterations executed by that invocation, including after resume. The audited [experiment workflow](docs/d2_experiments.md) enforces a total target and restores anomaly records; use it for guarded long runs. This refactor does not start long training or change protection thresholds.

Use the existing Hydra overrides, for example `agent.algorithm.num_mini_batches=8`. Logs/checkpoints remain under `logs/rsl_rl/`. Resume retains weights, both MoE Adam states and their learning rates, with iteration-based curricula continuing. Physics/rollout state is initialized anew. Space and resource checks in the experiment workflow remain unchanged.

## Playback, export and keyboard

```bash
./d2_rl_lab.sh --play --task d2-mixed-moe-cts --physics newton_mjwarp \
  --checkpoint /path/to/model.pt --num_envs 1 --viz none

./d2_rl_lab.sh --deploy-mujoco --config deploy/deploy_mujoco/d2.yaml \
  --policy /path/to/exported/policy.pt --terrain stairs --keyboard --duration 600

python deploy/deploy_mujoco/deploy.py --config deploy/deploy_mujoco/d2.yaml \
  --policy /path/to/exported/policy.pt --keyboard --duration 600
```

Playback exports TorchScript/ONNX alongside the selected run; MoE export uses the student. When copying checkpoints for export, also copy the original run's `params/` directory. Keep `config.yaml` beside exported files for identity/hash checks. TorchScript accepts one frame and holds history internally; ONNX accepts the history input. W/S changes forward speed, A/D lateral speed, Q/E yaw rate; **Space clears all velocity commands**. Viewer Reset resets simulation, history and delay. Use `--headless --no-real-time --no-joystick` for scripted batch replay.

Other existing CLI options remain: `--generate-vscode-settings`, `--train-distributed`, `--play --video`. Distributed training uses `CUDA_VISIBLE_DEVICES`. No additional command format is introduced. The old shell/Python entry names and Go2 import paths are removed; update custom scripts using [the migration table](docs/core_migration.md).

## Sources and validation

See [D2 source/license/hash records](docs/licenses/go2_rl_robotlab/NOTICE.txt), [experiment evaluation](docs/d2_experiments.md), and [migration](docs/core_migration.md). Original assets retain their notices; Kinatric-owned code and modifications use GPL-3.0-or-later, with upstream and vendored components retaining their own notices; see [LICENSING.md](LICENSING.md). Historical Go2 strings in provenance and model identity are deliberate, not live dependencies. RoboGauge has no default robot task; enabling it requires an explicitly supported evaluator task.

Local validation material stays in ignored `tests/upgrade_isaaclab_3_ea/`; logs, models, videos, environments and generated assets are not release content. Short regression checks establish interface compatibility, not stable walking or hardware readiness.

## License

Kinatric-owned software and firmware contributions are licensed under **GPL-3.0-or-later**. Kinatric-owned robot hardware designs and associated manufacturing documentation are licensed under **CERN-OHL-W-2.0**, subject to the file coverage in [LICENSING.md](LICENSING.md). Third-party material retains its original licenses and attribution; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Both full license texts are included in [LICENSES/](LICENSES/). Kinatric-owned trained policy weights are also GPL-3.0-or-later; see the coverage table and provenance requirements.
