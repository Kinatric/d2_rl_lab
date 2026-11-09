# SPDX-License-Identifier: Apache-2.0
# Adapted from lookc4/go2_rl_robotlab, commit 079b31694e11dead1ac9ff0ea0a60737eb8203c2.
"""Portable D2 asset and policy contracts, without simulator imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
DESCRIPTION = Path(__file__).resolve().parent / "d2_description"
RUNTIME = ROOT / "resources/d2_runtime"
MOTOR_PATH = DESCRIPTION / "config/robstride06_motor.yaml"
JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]
RL_POSE = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]
FIXSTAND_POSE = [0.0, 0.8, -1.5] * 4


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def motor_spec(path=MOTOR_PATH) -> dict:
    """Load the source motor contract; unsupported protection modes fail explicitly."""
    raw = yaml.safe_load(Path(path).read_text())
    if raw["protection_mode"] != "instantaneous_envelope":
        raise ValueError("D2 Lab 3.0 supports instantaneous_envelope only")
    curve = np.asarray(raw["peak_torque_speed_envelope_rad_s"], dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 2 or not np.isfinite(curve).all():
        raise ValueError("Invalid RS06 torque-speed curve")
    if curve[0, 0] != 0 or np.any(np.diff(curve[:, 0]) <= 0) or np.any(curve[:, 1] < 0):
        raise ValueError("Invalid RS06 torque-speed nodes")
    return {
        "curve": curve.tolist(),
        "peak": float(raw["output_limits"]["peak_torque_nm"]),
        "speed": float(raw["recommended_limit_layers"]["explicit_actuator_model"]["actuator_velocity_limit_rad_s"]),
        "armature": float(raw["mechanical"]["output_equivalent_inertia_kg_m2"]),
        "friction": float(raw["mechanical"]["output_joint_friction_nm"]),
    }


def checked_file(root: Path, item: dict) -> Path:
    path = (root / item["path"]).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("D2 manifest path escapes repository")
    if not path.is_file() or file_hash(path) != item["sha256"]:
        raise ValueError(f"D2 asset missing or changed: {path}")
    return path


def asset_manifest(require_ready=True) -> dict:
    """Verify source and generated dependencies before constructing a D2 runtime."""
    manifest = json.loads((RUNTIME / "asset_manifest.json").read_text())
    return validate_manifest(manifest, require_ready=require_ready)


def validate_manifest(manifest: dict, require_ready=True) -> dict:
    """Validate a candidate manifest before an atomic path-only migration."""
    if manifest.get("schema") != "robotlab.d2.runtime.v1":
        raise ValueError("Unsupported D2 manifest")
    for item in manifest["source"]["files"]:
        checked_file(DESCRIPTION, item)
    for name in ("urdf", "motor_yaml", "joint_adapter"):
        checked_file(RUNTIME, manifest["inputs"][name])
    for name in ("isaac_usd", "mujoco_scene"):
        output = manifest["outputs"][name]
        root = checked_file(RUNTIME, output).parent
        for item in output["files"]:
            checked_file(root, item)
    if manifest["joint_order"] != JOINT_NAMES:
        raise ValueError("D2 joint order differs")
    if require_ready:
        validation = manifest.get("validation", {})
        if validation.get("status") != "ready":
            raise ValueError("D2 assets need the three-backend standing validation")
        report = json.loads(checked_file(RUNTIME, validation["receipt"]).read_text())
        if report.get("status") != "passed" or report.get("runtime_identity") != runtime_identity(manifest):
            raise ValueError("D2 validation receipt is stale")
    return manifest


def runtime_identity(manifest: dict) -> str:
    return digest({"source": manifest["asset_identity_sha256"], "outputs": manifest["outputs"]})


def policy_identity(cfg=None, *, manifest=None) -> dict:
    """Bind the verified assets, controller and observation semantics."""
    manifest = asset_manifest() if manifest is None else manifest
    adapter = yaml.safe_load((DESCRIPTION / "config/joint_adapter.yaml").read_text())["simulation"]
    if (adapter["model_order"] != JOINT_NAMES or adapter["mujoco_order"] != JOINT_NAMES
            or adapter["direction_sign"] != [1]*12 or adapter["zero_offset_rad"] != [0.0]*12
            or adapter["mujoco_status"] != "commissioned"):
        raise ValueError("Unsupported D2 simulation joint adapter; explicit remapping is required")
    contract = {
        "schema": "go2_rl_lab.d2.policy.v1", "robot": "d2",
        "source_asset": manifest["asset_identity_sha256"], "runtime_asset": runtime_identity(manifest),
        "motor_sha256": file_hash(MOTOR_PATH), "joint_mapping_sha256": digest(adapter),
        "joint_names": JOINT_NAMES, "default_angles": RL_POSE,
        "physics_dt": 0.005, "decimation": 4, "kp": 25.0, "kd": 0.5,
        "delay_steps": [0, 4], "action_scale": 0.25,
        "raw_action_clip": None, "joint_target_clip": [-100.0, 100.0],
        "motor_parameters": motor_spec(),
        "command_kind": "command/body/velocity",
        "terms": ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"],
        "scales": [0.25, 1.0, [0.5, 1.0, 1.0], 1.0, 0.05, 1.0],
        "clip": [-100.0, 100.0], "history_length": 10,
    }
    if cfg is not None:
        obs = cfg.observations.policy
        contract["physics_dt"], contract["decimation"] = cfg.sim.dt, cfg.decimation
        contract["scales"] = [getattr(obs, term).scale for term in contract["terms"]]
        contract["history_length"] = obs.history_length
        if list(cfg.actions.joint_pos.joint_names) != JOINT_NAMES or not cfg.actions.joint_pos.preserve_order:
            raise ValueError("D2 action joint order differs from its simulation adapter")
        if not cfg.actions.joint_pos.use_default_offset:
            raise ValueError("D2 requires the configured RL joint pose as the action origin")
        contract["default_angles"] = [cfg.scene.robot.init_state.joint_pos[name] for name in JOINT_NAMES]
        contract["joint_target_clip"] = cfg.actions.joint_pos.clip[".*"]
        contract["observation_clips"] = [getattr(obs, term).clip for term in contract["terms"]]
        scale = cfg.actions.joint_pos.scale
        contract["action_scale"] = next(iter(scale.values())) if isinstance(scale, dict) and len(set(scale.values())) == 1 else scale
        actuator = cfg.scene.robot.actuators["RS06"]
        if Path(actuator.motor_config_path).resolve() != MOTOR_PATH.resolve():
            raise ValueError("D2 must use the verified canonical motor YAML")
        contract.update(kp=actuator.stiffness, kd=actuator.damping, delay_steps=[actuator.min_delay, actuator.max_delay])
        contract["motor_parameters"].update(peak=actuator.joint_effort_limit, speed=actuator.joint_velocity_limit,
                                            armature=actuator.armature, friction=actuator.friction)
    else:
        contract["observation_clips"] = [[-100.0, 100.0]] * 6
    # Normalize tuples so checkpoint and YAML round trips compare identically.
    return json.loads(json.dumps(contract))


def pose_height(pose) -> float:
    """Compute the lowest spherical foot from the calibrated URDF (source FK)."""
    root = ET.parse(DESCRIPTION / "urdf/d2_description.urdf").getroot()
    def transform(origin):
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        r, p, y = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        cr, cp, cy, sr, sp, sy = np.cos(r), np.cos(p), np.cos(y), np.sin(r), np.sin(p), np.sin(y)
        out = np.eye(4)
        out[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
        out[:3, 3] = xyz
        return out
    positions = dict(zip(JOINT_NAMES, pose, strict=True))
    transforms = {"base": np.eye(4)}
    pending = list(root.findall("joint"))
    while pending:
        progressed = False
        for joint in pending[:]:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in transforms:
                continue
            t = transforms[parent] @ transform(joint.find("origin"))
            if joint.get("type") == "revolute":
                a = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                a /= np.linalg.norm(a)
                x, y, z = a
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                q = positions[joint.get("name")]
                rotation = np.eye(4)
                rotation[:3, :3] = np.eye(3) + np.sin(q)*skew + (1-np.cos(q))*(skew @ skew)
                t = t @ rotation
            transforms[child] = t
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("D2 URDF is not a tree")
    bottoms = []
    for link in root.findall("link"):
        if link.get("name").endswith("_foot"):
            collision = link.find("collision")
            t = transforms[link.get("name")] @ transform(collision.find("origin"))
            bottoms.append(t[2, 3] - float(collision.find("geometry/sphere").get("radius")))
    if len(bottoms) != 4:
        raise ValueError("D2 requires four spherical feet")
    return float(-min(bottoms))
