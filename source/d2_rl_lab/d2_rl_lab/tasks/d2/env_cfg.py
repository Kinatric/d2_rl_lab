"""D2 variants preserve the Lab 3.0 training algorithms and observation contract."""

from isaaclab.utils.configclass import configclass
from d2_rl_lab.tasks.d2.flat_env_cfg import (
    FlatVelocityCommandCfg,
    FlatVelocityCommandsCfg,
    FlatVelocityRobotEnvCfg,
    FlatVelocityRobotPlayEnvCfg,
)
from d2_rl_lab.tasks.d2.mixed_env_cfg import RobotEnvCfg, RobotPlayEnvCfg


def configure_d2(cfg):
    from d2_rl_lab.assets.d2 import make_d2_cfg
    from d2_rl_lab.assets.d2_contract import FIXSTAND_POSE, pose_height
    from d2_rl_lab.tasks.d2.mdp import base_ground_contact
    from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
    cfg.scene.robot = make_d2_cfg()
    cfg.sim.use_newton_actuators = False
    height = pose_height(FIXSTAND_POSE)
    cfg.rewards.base_height_l2.params["target_height"] = height
    cfg.rewards.feet_regulation.params["base_height_target"] = height
    # Newton reports base/thigh self contacts in net_forces_w. Track terrain
    # shapes separately for the base-ground termination, retaining self collision.
    sensor = cfg.scene.contact_forces
    for variant in (sensor, getattr(sensor, "default", None), getattr(sensor, "newton_mjwarp", None)):
        if isinstance(variant, NewtonContactSensorCfg):
            variant.filter_shape_prim_expr = [cfg.scene.terrain.prim_path + "/.*"]
    cfg.terminations.illegal_contact.func = base_ground_contact
    cfg.policy_identity = cfg.get_policy_identity()


class D2PolicyContract:
    def get_policy_identity(self):
        from d2_rl_lab.assets.d2_contract import policy_identity
        return policy_identity(self)


@configclass
class D2FlatVelocityCommandCfg(FlatVelocityCommandCfg):
    """Include explicit standing samples without replacing them with yaw commands."""

    zero_command_curriculum: dict | None = {
        "start_iter": 0,
        "end_iter": 1500,
        "start_value": 0.0,
        "end_value": 0.1,
    }
    limit_ang_vel_at_zero_command_prob: float = 0.0


@configclass
class D2FlatCommandsCfg(FlatVelocityCommandsCfg):
    base_velocity: D2FlatVelocityCommandCfg = D2FlatVelocityCommandCfg()


@configclass
class D2FlatEnvCfg(D2PolicyContract, FlatVelocityRobotEnvCfg):
    commands: D2FlatCommandsCfg = D2FlatCommandsCfg()
    policy_identity: dict | None = None

    def __post_init__(self):
        super().__post_init__()
        configure_d2(self)


@configclass
class D2FlatPlayEnvCfg(D2PolicyContract, FlatVelocityRobotPlayEnvCfg):
    policy_identity: dict | None = None

    def __post_init__(self):
        super().__post_init__()
        configure_d2(self)


@configclass
class D2MixedEnvCfg(D2PolicyContract, RobotEnvCfg):
    policy_identity: dict | None = None

    def __post_init__(self):
        super().__post_init__()
        configure_d2(self)


@configclass
class D2MixedPlayEnvCfg(D2PolicyContract, RobotPlayEnvCfg):
    policy_identity: dict | None = None

    def __post_init__(self):
        super().__post_init__()
        configure_d2(self)
