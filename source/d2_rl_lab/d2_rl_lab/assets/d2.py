"""Construct D2 only when selected; task discovery does not require generated assets."""

from isaaclab.assets import ArticulationCfg
from robot_rl_core.assets.usd import RobotUsdFileCfg
from d2_rl_lab.actuators.robstride_actuator import RobStride06ActuatorCfg
from d2_rl_lab.assets.d2_contract import JOINT_NAMES, RL_POSE, FIXSTAND_POSE, RUNTIME, asset_manifest, motor_spec, pose_height


def make_d2_cfg(*, require_ready=True):
    manifest, motor = asset_manifest(require_ready), motor_spec()
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=RobotUsdFileCfg(usd_path=str(RUNTIME / manifest["outputs"]["isaac_usd"]["path"])),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, pose_height(FIXSTAND_POSE)+0.03),
            joint_pos=dict(zip(JOINT_NAMES, RL_POSE, strict=True)), joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={"RS06": RobStride06ActuatorCfg(
            joint_names_expr=JOINT_NAMES, stiffness=25.0, damping=0.5,
            joint_effort_limit=motor["peak"], joint_velocity_limit=motor["speed"],
            armature=motor["armature"], friction=motor["friction"], min_delay=0, max_delay=4,
        )},
    )
