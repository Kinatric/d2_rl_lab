# Repository guide

D2_rl_lab contains `robot_rl_core`, `d2_rl_lab` and one vendored `rsl_rl`. Go2 robot support and legacy command/import wrappers were explicitly removed. Keep upstream license, attribution and Git history, and keep historical policy identity data unchanged.

## Boundaries

- Never modify active production source, environments, models or anomaly records when validating a migration. Use a separate checkout and environment; do not change another job's editable install paths.
- Do not create commits, push, switch production or start long training unless explicitly requested.
- Keep all validation material under ignored `tests/upgrade_isaaclab_3_ea/`. Do not stage, commit or publish tests, logs, models, videos, incident snapshots, caches, environments or generated USD/runtime assets. User-authorized transfer to an isolated validation host is allowed; it is not publication.
- Reusable source workflows belong in `scripts/d2_experiments/` and `scripts/d2_asset_checks/`; never copy credentials, results or experiment-specific paths into them.
- Before an authorized commit/push, verify `git ls-files tests` is empty, scan release content, and preserve original asset licenses/hashes. There is no bundled policy exception anymore.

Current original assets are stored as ordinary Git blobs, not Git LFS pointers. Preserve their bytes and hashes when publishing. Historical LFS objects remain available for old revisions; do not rewrite history or remove those objects. This does not permit publishing models, tests or generated runtime assets.

## Architecture

| Path | Responsibility |
| --- | --- |
| `d2_rl_lab.sh` | Original CLI flags and operation flow |
| `source/robot_rl_core/robot_rl_core/` | Robot-independent commands, MDP functions, terrain/physics/asset utilities and task discovery |
| `source/d2_rl_lab/d2_rl_lab/` | D2 assets, motor, identity, observation/action/environment/network configuration and symmetry mapping |
| `source/rsl_rl/rsl_rl/` | Shared PPO/MoE-CTS algorithms, storage, runners and exports |
| `scripts/rsl_rl/` | Runtime bootstrap, train/play/configuration/video orchestration |
| `deploy/deploy_mujoco/deploy.py` | Common simulation, keyboard, render loop; robot behavior selected by existing YAML `robot` field |

Three tasks only: `d2-flat`, `d2-mixed`, `d2-mixed-moe-cts`. Gym registration, class inheritance and Hydra remain the task mechanism. D2-specific 12 joints, 45-frame/10-history layout and RS06 mapping must not become common-core constants. Future robots depend on the core, not D2; do not invent a plugin system or register unimplemented D2W tasks.

## Runtime

Python 3.12, Isaac Lab 3.0, Isaac Sim 6.2, PyTorch 2.11/CUDA 12.8, Newton 1.6.1 and Warp 1.17. Inspect versions when changing environments. Local RSL-RL distribution is 3.3.0 with project extensions; do not let upstream dependency upgrades replace it.

```bash
./d2_rl_lab.sh --install
./d2_rl_lab.sh --list
python -m pip install --no-deps -e source/rsl_rl -e source/robot_rl_core -e source/d2_rl_lab
```

Bootstrap before USD imports; never mix Kit and standalone USD in one process. Resolve registered configs through `isaaclab_tasks.utils.resolve_task_config`, restoring temporary registry changes in `finally`. Use `newton_mjwarp` / `isaacsim_physx` and `--viz none`, not obsolete training `--headless`. Standalone MuJoCo retains its own `--headless`.

Isaac Lab data uses ProxyArray `.torch`/`.warp`, quaternions XYZW, selected updates use keyword-only indexed APIs. Keep RS06's Python actuator path (`sim.use_newton_actuators=False`), inertia/friction applied once, self-collision, fixed feet, inertial principal axes and joint velocity limits. Preserve terrain-filtered Newton base-ground termination. Do not change rewards, curricula, observations, action scaling, motor parameters or recovery thresholds as part of structural cleanup.

Preserve identity checks. Old state-dict checkpoints and exports remain usable without old code paths; arbitrary old config pickles are not a reason to recreate the removed package. Normal resume restores both MoE Adam states, learning rates and iteration-based curricula; audited workflows also restore anomaly ledgers. Simulation state initializes anew.

## Validation

```bash
git diff --check
bash -n d2_rl_lab.sh
python -m compileall -q scripts source deploy
./d2_rl_lab.sh --train --task d2-flat --physics newton_mjwarp \
  --viz none --num_envs 32 --max_iterations 2 --logger tensorboard
```

For shared changes run all three D2 tasks on both backends, serially. Verify real rollout, finite loss/model/optimizer state, actor/critic and teacher/student expert/gate updates and readable checkpoints. Exit code alone is insufficient; Kit may exit before training. Check old PPO/MoE resume, curricula and both optimizer states, asset/motor/guard tests, student TorchScript/ONNX and identical-model MuJoCo stand/walk-stop-walk trajectories. Preserve Space-to-stop and reset behavior. Resource thresholds: 2 GiB GPU, 5 GiB instance RAM, 10 GiB disk; wait if insufficient, never stop unrelated workloads.

Validation evidence is local and separate from source. Distinguish passing short interface checks from walking convergence, D2W support, interactive GUI, multi-GPU and production-scale validation. Keep documentation and import migration instructions synchronized. Maintain four-space Python indentation and concise runnable examples.
