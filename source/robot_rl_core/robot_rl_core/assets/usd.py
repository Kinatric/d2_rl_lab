"""Isaac Lab articulation configuration for the robot assets."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass


if TYPE_CHECKING:
    from pxr import Usd

def _deactivate_visual_branches(root_prim: Usd.Prim) -> int:
    """Deactivate normalized robot ``visuals`` branches before environment replication.
    We found that this can very effectively reduce cuda memory usage.
    """
    from pxr import Usd

    stage = root_prim.GetStage()
    visual_paths = [prim.GetPath() for prim in Usd.PrimRange(root_prim) if prim.GetName() == "visuals"]
    for path in visual_paths:
        stage.GetPrimAtPath(path).SetActive(False)
    return len(visual_paths)


def spawn_robot_from_usd(
    prim_path: str,
    cfg: RobotUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn robot and optionally remove its visual-only branches before scene cloning."""
    robot_prim = sim_utils.spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    if not cfg.visuals_enabled:
        _deactivate_visual_branches(robot_prim)
    return robot_prim


@configclass
class RobotUsdFileCfg(sim_utils.UsdFileCfg):
    """USD spawn configuration used by the robot articulation."""

    func: Callable | str = spawn_robot_from_usd
    visuals_enabled: bool = True
    """Whether visual-only branches remain active when the source robot is cloned."""

    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
