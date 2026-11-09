"""D2 deployment adapter using the same physical contract as training."""

from collections import deque
import json
from pathlib import Path
import numpy as np
import yaml

from d2_rl_lab.assets.d2_contract import (
    FIXSTAND_POSE, JOINT_NAMES, RL_POSE, RUNTIME, asset_manifest,
    file_hash, motor_spec, policy_identity, pose_height,
)


class PositionDelay:
    """Match Isaac Lab DelayBuffer's first-value padding and per-reset lag."""
    def __init__(self, low=0, high=4, seed=0):
        self.low, self.high = low, high
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.lag = int(self.rng.integers(self.low, self.high+1))
        self.values = deque(maxlen=self.high+1)

    def compute(self, target):
        if not self.values:
            self.values.extend(target.copy() for _ in range(self.high+1))
        else:
            self.values.append(target.copy())
        return self.values[-self.lag-1]


def prepare_d2(config, policy_path: Path):
    """Check the policy and populate generated paths from the verified manifest."""
    manifest = asset_manifest()
    metadata = yaml.safe_load(policy_path.with_name("config.yaml").read_text())
    expected = policy_identity(manifest=manifest)
    if metadata.get("policy_identity") != expected:
        raise ValueError("D2 policy asset/controller/observation identity mismatch")
    if metadata.get("policy_files", {}).get(policy_path.name) != file_hash(policy_path):
        raise ValueError("D2 export hash differs from config.yaml")
    checks = {
        "default_angles": RL_POSE, "policy_joint_names": JOINT_NAMES, "mujoco_joint_names": JOINT_NAMES,
        "command_kind": expected["command_kind"], "command_scale": expected["scales"][2],
        "ang_vel_scale": 0.25, "dof_pos_scale": 1.0, "dof_vel_scale": 0.05,
        "action_scale": expected["action_scale"], "action_clip": expected["raw_action_clip"],
        "joint_target_clip": expected["joint_target_clip"],
        "num_actions": 12, "num_obs": 45, "simulation_dt": 0.005, "control_decimation": 4,
        "kps": [25.0]*12, "kds": [0.5]*12,
    }
    for key, value in checks.items():
        if config.get(key) != value:
            raise ValueError(f"D2 deployment {key} differs from the policy contract")
    scene = str(RUNTIME / manifest["outputs"]["mujoco_scene"]["path"])
    config["terrains"] = {"flat": scene, "stairs": scene}
    config["motor"] = {"type": "rs06", **motor_spec()}
    config["base_height"] = pose_height(FIXSTAND_POSE)+0.03
    return PositionDelay(*expected["delay_steps"], seed=0)


def clip_rs06(effort, velocity, motor):
    curve = np.asarray(motor["curve"])
    limit = np.interp(np.abs(velocity), curve[:, 0], curve[:, 1], right=0.0)
    return np.clip(effort, -limit, limit)


def d2_observation(angular_velocity, gravity, command, position, velocity, previous_action):
    """Match manager term clipping before scaling and preserve the raw last action."""
    terms = [angular_velocity, gravity, command, position-np.asarray(RL_POSE), velocity, previous_action]
    scales = [0.25, 1.0, np.array([0.5, 1.0, 1.0]), 1.0, 0.05, 1.0]
    return np.concatenate([np.clip(term, -100., 100.)*scale for term,scale in zip(terms,scales)]).astype(np.float32)


class D2MujocoAdapter:
    """D2-specific hooks used by the common deployment loop."""

    tracking_body = "base"
    visual_urdf_name = "d2_description.urdf"

    def __init__(self, config, policy_path, explicit_xml=None):
        self.config = config
        self.delay = prepare_d2(config, policy_path)
        if explicit_xml is not None:
            raise ValueError("D2 uses the validated scene selected by its asset manifest")

    def reset(self, data, *, initial=False):
        data.qpos[2] = self.config["base_height"]
        if not initial:
            self.delay.reset()

    def effort(self, target, position, stiffness, velocity, damping):
        effort = (self.delay.compute(target) - position) * stiffness - velocity * damping
        return clip_rs06(effort, velocity, self.config["motor"])

    def observe(self, data, gravity, command, position, velocity, previous_action):
        return d2_observation(data.qvel[3:6], gravity, command, position, velocity, previous_action)

    def action_target(self, action, default_position):
        target = default_position + action * float(self.config["action_scale"])
        return np.clip(target, *self.config["joint_target_clip"])
