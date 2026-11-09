# Public core migration / 公共核心迁移

The checkout now contains `robot_rl_core`, `d2_rl_lab` and one `rsl_rl`. Go2 robot assets, actuator, tasks, demo policy, old package and command wrappers have been removed. Upstream history and license notices remain. This is a source/entry-point migration; D2 reward weights, curricula, observations, normalization, actions, motor curves, delays and anomaly thresholds are unchanged.

当前仅有公共核心、D2 包和唯一一份 RSL-RL。Go2 机器人、任务、电机、演示模型、旧包和旧入口已移除；Git 历史、许可证和来源保留。D2 的奖励、课程、观测、归一化、动作、电机、延迟及异常阈值不变。

| Previous path / 旧路径 | Current path / 新路径 |
| --- | --- |
| `go2_rl_lab.sh` | `d2_rl_lab.sh` |
| `deploy/deploy_mujoco/deploy_go2.py` | `deploy/deploy_mujoco/deploy.py` |
| `go2_rl_lab.cmd_utils`, `.physics`, `.utils` | `robot_rl_core.cmd_utils`, `.physics`, `.utils` |
| `go2_rl_lab.tasks.mixed.mdp` | `robot_rl_core.mdp` |
| `go2_rl_lab.tasks.mixed.terrain` | `robot_rl_core.utils.terrains.mixed` |
| `go2_rl_lab.tasks.mixed.terrain_mesh` | `robot_rl_core.utils.terrains.terrain_mesh` |
| `go2_rl_lab.assets.go2:Go2UsdFileCfg` | `robot_rl_core.assets.usd:RobotUsdFileCfg` |
| `go2_rl_lab.tasks.mixed.env_cfg` | `d2_rl_lab.tasks.d2.mixed_env_cfg` |
| `go2_rl_lab.tasks.flat.env_cfg` | `d2_rl_lab.tasks.d2.flat_env_cfg` |
| `go2_rl_lab.tasks.mixed.rsl_rl_cfg:Go2PPORunnerCfg/Go2MoECTSRunnerCfg` | `d2_rl_lab.tasks.d2.rsl_rl_cfg:D2PPORunnerCfg/D2MoECTSRunnerCfg` |
| `go2_rl_lab.utils.symmetry:Go2Symmetry/Go2SymmetryMapper` | `d2_rl_lab.symmetry:D2Symmetry/D2SymmetryMapper` |
| legacy `go2_rl_lab.assets.d2`, `.d2_contract`, `.robstride_actuator` | `d2_rl_lab.assets.d2`, `.d2_contract`, `d2_rl_lab.actuators.robstride_actuator` |
| legacy `go2_rl_lab.tasks.d2` | `d2_rl_lab.tasks.d2` |
| deployment `utils.d2` | `d2_rl_lab.deployment.mujoco` |
| `GO2_MIXED_TERRAIN` | `D2_MIXED_TERRAIN` |
| policy path placeholder `{GO2_RL_LAB_ROOT_DIR}` | `{D2_RL_LAB_ROOT_DIR}` |

No old import or command forwarding is retained. Update custom scripts and saved YAML Python callable paths using this table. Raw state-dict checkpoints and exported policies do not require old Python modules. Arbitrary pickled old Python config objects are not supported by resurrecting deleted packages.

不保留旧导入或命令转发。自定义脚本和旧 YAML 中的 Python 函数路径按上表更新。PPO/MoE 权重及导出模型无需旧 Python 包；任意旧配置对象的 pickle 不通过恢复已删除的旧包来兼容。

## Assets and old models / 资产与旧模型

The description remains at `source/d2_rl_lab/d2_rl_lab/assets/d2_description`; its bytes and hashes are unchanged. Copy existing `resources/d2_runtime/` into a separate checkout, including the manifest and receipts. For a pre-dual-package manifest that still points at `resources/d2_description`, run `python scripts/tools/migrate_d2_asset_paths.py`. It validates all source/output hashes and identity before changing only input paths. Already migrated manifests need no changes. Fresh clones follow [asset generation](d2_assets.md).

原始资产位置与上一版双包结构一致，内容及哈希不变。复制旧运行资产及清单、验证凭据至独立副本。更早版本的清单仍指向 `resources/d2_description` 时，使用上述迁移脚本；已迁移的清单无需改动。失败必须定位，不关闭校验。

Keep checkpoint and export metadata unchanged. Historical data identifiers `go2_rl_lab.d2.policy.v1` and `go2-rl-lab3-d2-v1` remain deliberately: they are identity data, not executable import paths. Resume still uses `--resume --checkpoint`; MoE restores both Adam states. The main optimizer's restored learning rate is also synchronized to the adaptive schedule, matching the existing audited training workflow. Iteration-based curricula continue; physics and policy rollout history initialize anew. The audit workflow restores anomaly ledgers; plain CLI does not replace that workflow.

旧模型及元数据不改写，历史身份字符串作为数据保留。MoE 恢复两套 Adam，主学习率同步到自适应调度变量，等同既有审计训练行为。课程接续，物理现场与采样历史重新初始化，不保证逐轨迹复现。异常账本仍通过原审计工作流恢复；普通 CLI 不替代该工作流。

## Future robots / 后续机器人

A future `d2w_rl_lab` depends on `robot_rl_core` and `rsl_rl`, not D2. Define its assets, motor, observation/action configuration and symmetry mapping in that package. Add its package to the same `register_tasks` discovery list and select its adapter through the existing MuJoCo YAML `robot` field. No plugin system, new configuration format, D2W package or D2W task is introduced here. Common augmentation has no fixed joint/foot count; the deployment loop requests robot-specific observation, effort, reset and action-target hooks.

后续 D2W 使用相同 Gym、配置继承、Hydra 和 MuJoCo YAML 机制；不依赖 D2，也不复制算法。本轮只验证不同维度的测试适配器，不表示 D2W 已接入。

Install in a separate environment: `python -m pip install --no-deps -e source/rsl_rl -e source/robot_rl_core -e source/d2_rl_lab`. Never redirect editable installs used by active jobs. Production switching, commits and publication require a separate instruction.
