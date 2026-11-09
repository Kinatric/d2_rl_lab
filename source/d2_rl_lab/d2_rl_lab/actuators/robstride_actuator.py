"""RS06 delayed explicit PD actuator for Isaac Lab 3.0."""

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils.configclass import configclass
from d2_rl_lab.assets.d2_contract import MOTOR_PATH, motor_spec


def torque_limit(speed, nodes):
    """Interpolate the symmetric output-side peak envelope, including overspeed."""
    speed = speed.abs()
    upper = torch.searchsorted(nodes[:, 0].contiguous(), speed.contiguous(), right=True).clamp(1, len(nodes)-1)
    lower = upper - 1
    fraction = ((speed - nodes[lower, 0]) / (nodes[upper, 0] - nodes[lower, 0])).clamp(0, 1)
    return (nodes[lower, 1] + fraction*(nodes[upper, 1]-nodes[lower, 1])).clamp(min=0)


class RobStride06Actuator(DelayedPDActuator):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.nodes = torch.tensor(motor_spec(cfg.motor_config_path)["curve"], device=self._device)

    def _clip_effort(self, effort, joint_vel):
        limit = torque_limit(joint_vel, self.nodes)
        return torch.clamp(effort, -limit, limit)


@configclass
class RobStride06ActuatorCfg(DelayedPDActuatorCfg):
    class_type: type = RobStride06Actuator
    motor_config_path: str = str(MOTOR_PATH)
    joint_effort_limit: float = 36.0
    joint_velocity_limit: float = 50.0
