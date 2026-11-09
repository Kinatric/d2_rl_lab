# D2 asset build and validation / 资产构建与验证

Run from the repository root in the installed Isaac Lab 3.0 environment. All generated assets and receipts stay under ignored `resources/d2_runtime`; validation evidence stays under ignored `tests/upgrade_isaaclab_3_ea/d2_asset_validation`. No old receipt is shipped.

在仓库根目录、已安装的 Isaac Lab 3.0 环境执行。构建保留 17 个连杆、12 个转动关节、4 个固定足端、25 个碰撞体及 16.1086 kg 标定总质量；来源和哈希见 `docs/licenses/go2_rl_robotlab/`。不会随仓库发布旧验证凭据。

```bash
python scripts/tools/build_d2_runtime.py prepare
python - <<'PY'
import json, subprocess, sys
from pathlib import Path
s = json.loads(Path('resources/d2_runtime/.build_state.json').read_text())
subprocess.run([sys.executable, 'scripts/tools/convert_urdf.py',
                s['usd_urdf'], s['usd_output'], '--no-merge-fixed-joints', '--viz', 'none'], check=True)
PY
python scripts/tools/build_d2_runtime.py finalize
python scripts/d2_asset_checks/standing.py --backend mujoco --viz none
python scripts/d2_asset_checks/standing.py --backend newton_mjwarp --viz none
python scripts/d2_asset_checks/standing.py --backend isaacsim_physx --viz none
python scripts/d2_asset_checks/promote.py
python scripts/d2_asset_checks/contracts.py
```

The three standing checks run serially for 30 simulated seconds each and validate geometry/dynamics. Promotion requires all three matching runtime identities and passing reports; rebuilding invalidates the receipt. A failed check must be diagnosed, not bypassed. PhysX/Kit requires a supported driver; do not replace a cloud host driver inside its container.

三后端站立验证各运行 30 秒仿真时间并串行执行。只有三份报告通过且资产身份一致才生成训练准入凭据；重建后需重新验证。`contracts.py` 检查扭矩、延迟、映射和身份一致性。失败保留日志并定位原因，不跳过准入检查；PhysX/Kit 驱动问题由宿主平台处理。

Reproduction preserves configuration and checks, not bitwise trajectories across backends or versions. Passing asset checks does not establish a trained walking policy or hardware readiness.

复现指配置和验收流程可重跑，不保证不同后端、版本间逐轨迹一致；资产验证通过不等于策略已学会行走或可以上实机。

Source description: `source/d2_rl_lab/d2_rl_lab/assets/d2_description`. For existing generated assets, follow [path-only migration](core_migration.md) instead of rebuilding.

原始描述包已迁入 D2 独立包；已有运行资产按[路径迁移说明](core_migration.md)处理，无须因目录变化重新转换。
