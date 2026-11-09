#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Adapted from lookc4/go2_rl_robotlab, commit 079b31694e11dead1ac9ff0ea0a60737eb8203c2.
"""Build the canonical D2 runtime package from source/d2_rl_lab/d2_rl_lab/assets/d2_description.

The source package is a controlled derivative of the pinned upstream archive:
its measured mass calibration is declared below and becomes part of the asset
identity.  All copied/imported files are written under the content-addressed
resources/d2_runtime/<asset-id>/ directory and described by one fail-closed
manifest at resources/d2_runtime/asset_manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml


EXPECTED_ARCHIVE_SHA256 = "645ddabe5690280b8a323c2c6d05d0061d70a6efc40ab712e6c617ed0295ff69"
EXPECTED_TOPOLOGY = {"links": 17, "revolute_joints": 12, "fixed_joints": 4, "collisions": 25}
MASS_PROPERTIES = {
    "schema": "robotlab.d2.measured-mass.v1",
    "method": "scale_full_inertia_tensor_by_new_mass_over_old_mass_preserving_com_geometry_and_inertial_frame",
    "upstream_archive_sha256": EXPECTED_ARCHIVE_SHA256,
    "units": {"mass": "kg", "inertia": "kg*m^2"},
    "link_types": {
        "base": {"old_mass_kg": 5.0, "new_mass_kg": 5.425, "inertia_scale": 1.085},
        "hip": {"old_mass_kg": 0.788, "new_mass_kg": 0.7737, "inertia_scale": 0.7737 / 0.788},
        "thigh": {"old_mass_kg": 1.45, "new_mass_kg": 1.6266, "inertia_scale": 1.6266 / 1.45},
        "calf": {"old_mass_kg": 0.23, "new_mass_kg": 0.2013, "inertia_scale": 0.2013 / 0.23},
        "foot": {"old_mass_kg": 0.02, "new_mass_kg": 0.0693, "inertia_scale": 0.0693 / 0.02},
    },
    "lower_leg_assembly": {
        "calf_mass_kg": 0.2013,
        "foot_mass_kg": 0.0693,
        "calf_plus_foot_mass_kg": 0.2706,
    },
    "per_leg_mass_kg": 2.6709,
    "total_mass_kg": 16.1086,
}
JOINT_ORDER = [
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
]
FIXED_FOOT_JOINTS = [f"{leg}_foot_joint" for leg in ("FL", "FR", "RL", "RR")]
SOURCE_MEMBERS = [
    "CMakeLists.txt",
    "package.xml",
    "config/joint_adapter.yaml",
    "config/robstride06_motor.yaml",
    "launch/check_joint.rviz",
    "launch/d2_rviz.launch.py",
    "urdf/d2_description.urdf",
    "meshes/base.stl",
    *(f"meshes/{leg}_{joint}.stl" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def entry(path: Path, relative_to: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def expected_link_mass(link_name: str) -> float:
    kind = "base" if link_name == "base" else link_name.rsplit("_", 1)[-1]
    try:
        return float(MASS_PROPERTIES["link_types"][kind]["new_mass_kg"])
    except KeyError as exc:
        raise RuntimeError(f"D2 mass contract has no entry for link {link_name!r}.") from exc


def rotation_from_rpy(rpy: list[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def apply_urdf_inertias_to_mjcf(urdf_root: ET.Element, mjcf_root: ET.Element) -> None:
    """Write exact URDF inertias into MJCF body coordinates.

    MuJoCo's URDF importer preserves mass and COM but treats the six inertia
    coefficients as link-frame values.  URDF defines them in the rotated
    inertial frame, so explicitly rotate each tensor and use MJCF fullinertia.
    """
    bodies = {body.get("name"): body for body in mjcf_root.findall(".//body")}
    for link in urdf_root.findall("link"):
        name = str(link.get("name"))
        body = bodies.get(name)
        inertial = link.find("inertial")
        mjcf_inertial = None if body is None else body.find("inertial")
        if inertial is None or mjcf_inertial is None:
            raise RuntimeError(f"Cannot map URDF inertia to MuJoCo body: {name}")
        origin = inertial.find("origin")
        inertia = inertial.find("inertia")
        if origin is None or inertia is None:
            raise RuntimeError(f"URDF inertial data is incomplete: {name}")
        xyz = [float(value) for value in origin.get("xyz", "0 0 0").split()]
        rpy = [float(value) for value in origin.get("rpy", "0 0 0").split()]
        tensor = np.asarray([
            [float(inertia.get("ixx")), float(inertia.get("ixy")), float(inertia.get("ixz"))],
            [float(inertia.get("ixy")), float(inertia.get("iyy")), float(inertia.get("iyz"))],
            [float(inertia.get("ixz")), float(inertia.get("iyz")), float(inertia.get("izz"))],
        ])
        rotation = rotation_from_rpy(rpy)
        tensor = rotation @ tensor @ rotation.T
        mjcf_inertial.set("pos", " ".join(format(value, ".17g") for value in xyz))
        mjcf_inertial.set("fullinertia", " ".join(format(value, ".17g") for value in (
            tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2],
        )))
        mjcf_inertial.attrib.pop("quat", None)
        mjcf_inertial.attrib.pop("diaginertia", None)


def validate_source(source: Path, archive: Path | None) -> tuple[ET.Element, list[dict[str, object]]]:
    expected = set(SOURCE_MEMBERS)
    actual = {path.relative_to(source).as_posix() for path in source.rglob("*")
              if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    if actual != expected:
        raise RuntimeError(f"D2 source package inventory differs: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    if archive is not None:
        actual_archive = sha256_file(archive)
        if actual_archive != EXPECTED_ARCHIVE_SHA256:
            raise RuntimeError(f"D2 source ZIP hash differs: expected={EXPECTED_ARCHIVE_SHA256}, actual={actual_archive}")
    root = ET.parse(source / "urdf/d2_description.urdf").getroot()
    if root.tag != "robot" or root.get("name") != "d2":
        raise RuntimeError("D2 URDF must contain <robot name='d2'>.")
    joints = root.findall("joint")
    topology = {
        "links": len(root.findall("link")),
        "revolute_joints": sum(j.get("type") == "revolute" for j in joints),
        "fixed_joints": sum(j.get("type") == "fixed" for j in joints),
        "collisions": len(root.findall(".//collision")),
    }
    if topology != EXPECTED_TOPOLOGY:
        raise RuntimeError(f"D2 URDF topology differs: expected={EXPECTED_TOPOLOGY}, actual={topology}")
    revolute = [j.get("name") for j in joints if j.get("type") == "revolute"]
    fixed = [j.get("name") for j in joints if j.get("type") == "fixed"]
    if revolute != JOINT_ORDER or fixed != FIXED_FOOT_JOINTS:
        raise RuntimeError(f"D2 joint order differs: revolute={revolute}, fixed={fixed}")
    total_mass = 0.0
    for link in root.findall("link"):
        inertial = link.find("inertial")
        mass = None if inertial is None else inertial.find("mass")
        if mass is None:
            raise RuntimeError(f"D2 link has no mass: {link.get('name')}")
        actual_mass = float(mass.get("value"))
        expected_mass = expected_link_mass(str(link.get("name")))
        if abs(actual_mass - expected_mass) > 1.0e-12:
            raise RuntimeError(
                f"D2 measured mass differs for {link.get('name')}: "
                f"expected={expected_mass}, actual={actual_mass}"
            )
        total_mass += actual_mass
    if abs(total_mass - float(MASS_PROPERTIES["total_mass_kg"])) > 1.0e-12:
        raise RuntimeError(f"D2 total mass differs: expected={MASS_PROPERTIES['total_mass_kg']}, actual={total_mass}")
    for mesh in root.findall(".//mesh"):
        name = mesh.get("filename")
        if not name or not (source / "urdf" / name).resolve().is_file():
            raise RuntimeError(f"D2 URDF mesh cannot be resolved: {name!r}")
    return root, [entry(source / name, source) for name in sorted(expected)]


def prepare(repo: Path, archive: Path | None) -> Path:
    source = repo / "source/d2_rl_lab/d2_rl_lab/assets/d2_description"
    _, source_files = validate_source(source, archive)
    identity_payload = {
        "schema": "robotlab.d2.source.v2",
        "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "mass_properties": MASS_PROPERTIES,
        "files": source_files,
        "topology": EXPECTED_TOPOLOGY,
        "joint_order": JOINT_ORDER,
        "fixed_foot_joints": FIXED_FOOT_JOINTS,
    }
    asset_id = canonical_digest(identity_payload)
    package = repo / "resources/d2_runtime" / asset_id[:12]
    if package.exists():
        shutil.rmtree(package)
    for relative_root in ("usd", "mujoco"):
        target = package / relative_root
        (target / "urdf").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "urdf/d2_description.urdf", target / "urdf/d2_description.urdf")
        shutil.copytree(source / "meshes", target / "meshes")

    # MuJoCo accepts URDF directly.  Saving the compiled model gives a stable
    # MJCF representation while retaining a copy of every dependency locally.
    import mujoco

    staged_urdf = package / "mujoco/urdf/d2_description.urdf"
    # MuJoCo's URDF compiler otherwise fuses the inertial base and fixed feet
    # into the world/adjacent bodies.  This compiler extension is added only
    # to the generated import copy; the authoritative ROS package is untouched.
    staged_tree = ET.parse(staged_urdf)
    extension = ET.SubElement(staged_tree.getroot(), "mujoco")
    ET.SubElement(extension, "compiler", {"fusestatic": "false"})
    staged_tree.write(staged_urdf, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(staged_urdf))
    model_xml = package / "mujoco/d2.xml"
    mujoco.mj_saveLastXML(str(model_xml), model)
    model_tree = ET.parse(model_xml)
    model_root = model_tree.getroot()
    apply_urdf_inertias_to_mjcf(staged_tree.getroot(), model_root)
    option = model_root.find("option")
    if option is None:
        option = ET.Element("option")
        model_root.insert(1, option)
    option.set("timestep", "0.005")
    option.set("gravity", "0 0 -9.81")
    base = model_root.find("./worldbody/body[@name='base']")
    if base is None:
        raise RuntimeError("MuJoCo URDF conversion did not preserve the D2 base body.")
    base.insert(0, ET.Element("freejoint", {"name": "base_free"}))
    spec = yaml.safe_load((source / "config/robstride06_motor.yaml").read_text())
    actuators = ET.SubElement(model_root, "actuator")
    for joint in model_root.findall(".//joint"):
        if joint.get("name") in JOINT_ORDER:
            joint.set("armature", str(spec["mechanical"]["output_equivalent_inertia_kg_m2"]))
            joint.set("frictionloss", str(spec["mechanical"]["output_joint_friction_nm"]))
            joint.set("damping", "0")
            ET.SubElement(actuators, "motor", {"name": joint.get("name")+"_motor", "joint": joint.get("name"), "gear": "1", "ctrllimited": "true", "ctrlrange": "-36 36"})
    # Keep the immutable asset source separate from the regenerated Lab 3.0 model.
    model_root.find("compiler").set("meshdir", "meshes")
    model_tree.write(model_xml, encoding="utf-8", xml_declaration=True)
    compiled = mujoco.MjModel.from_xml_path(str(model_xml))
    if compiled.njnt != 13 or compiled.nbody != 18:
        raise RuntimeError(
            f"Generated D2 MJCF lost its floating base/fixed feet: njnt={compiled.njnt}, nbody={compiled.nbody}"
        )
    scene_root = ET.parse(model_xml).getroot()
    world = scene_root.find("worldbody")
    if world is None:
        raise RuntimeError("Generated D2 MJCF has no worldbody.")
    ET.SubElement(world, "geom", {
        "name": "ground", "type": "plane", "size": "5 5 0.1",
        "rgba": "0.25 0.25 0.25 1", "friction": "1.0 0.005 0.0001",
    })
    ET.SubElement(world, "light", {"name": "sun", "pos": "0 0 3", "dir": "0 0 -1"})
    scene_xml = package / "mujoco/scene.xml"
    ET.ElementTree(scene_root).write(scene_xml, encoding="utf-8", xml_declaration=True)

    build_state = {
        "schema": "robotlab.d2.runtime-build.v1",
        "asset_identity_sha256": asset_id,
        "source_identity": identity_payload,
        "package": package.relative_to(repo / "resources/d2_runtime").as_posix(),
        "usd_urdf": (package / "usd/urdf/d2_description.urdf").relative_to(repo).as_posix(),
        "usd_output": (package / "usd/d2.usd").relative_to(repo).as_posix(),
    }
    state_path = repo / "resources/d2_runtime/.build_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(build_state, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(build_state, ensure_ascii=False, indent=2))
    return package


def finalize(repo: Path) -> Path:
    runtime = repo / "resources/d2_runtime"
    state_path = runtime / ".build_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    package = runtime / state["package"]
    usd_path = repo / state["usd_output"]
    if not usd_path.is_file():
        raise RuntimeError(f"Isaac USD output is missing: {usd_path}")
    usd_root = usd_path.parent
    usd_files = [entry(path, usd_root) for path in sorted(usd_root.rglob("*")) if path.is_file()]
    mujoco_root = package / "mujoco"
    mujoco_files = [entry(path, mujoco_root) for path in sorted(mujoco_root.rglob("*")) if path.is_file()]
    description = repo / "source/d2_rl_lab/d2_rl_lab/assets/d2_description"
    source_files = state["source_identity"]["files"]
    manifest = {
        "schema": "robotlab.d2.runtime.v1",
        "schema_version": 1,
        "asset_identity_sha256": state["asset_identity_sha256"],
        "canonical_asset_sha256": state["asset_identity_sha256"],
        "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "mass_properties": MASS_PROPERTIES,
        "source": {"root": "../../source/d2_rl_lab/d2_rl_lab/assets/d2_description", "files": source_files},
        "topology": {**EXPECTED_TOPOLOGY, "joint_order": JOINT_ORDER, "fixed_foot_joints": FIXED_FOOT_JOINTS},
        "joint_order": JOINT_ORDER,
        "control": {"physics_dt": 0.005, "decimation": 4, "policy_hz": 50.0},
        "generation": {"generator_version": "go2-rl-lab3-d2-v1", "fixed_joints_merged": False},
        "inputs": {
            "urdf": {"path": "../../source/d2_rl_lab/d2_rl_lab/assets/d2_description/urdf/d2_description.urdf", "sha256": sha256_file(description / "urdf/d2_description.urdf")},
            "motor_yaml": {"path": "../../source/d2_rl_lab/d2_rl_lab/assets/d2_description/config/robstride06_motor.yaml", "sha256": sha256_file(description / "config/robstride06_motor.yaml")},
            "joint_adapter": {"path": "../../source/d2_rl_lab/d2_rl_lab/assets/d2_description/config/joint_adapter.yaml", "sha256": sha256_file(description / "config/joint_adapter.yaml")},
            "meshes": [item for item in source_files if str(item["path"]).startswith("meshes/")],
        },
        "outputs": {
            "isaac_usd": {
                "status": "ready", "path": usd_path.relative_to(runtime).as_posix(),
                "sha256": sha256_file(usd_path), "files": usd_files, "tree_sha256": canonical_digest(usd_files),
            },
            "mujoco_xml": {
                "status": "ready", "path": (package / "mujoco/d2.xml").relative_to(runtime).as_posix(),
                "sha256": sha256_file(package / "mujoco/d2.xml"),
            },
            "mujoco_scene": {
                "status": "ready", "path": (package / "mujoco/scene.xml").relative_to(runtime).as_posix(),
                "sha256": sha256_file(package / "mujoco/scene.xml"), "files": mujoco_files,
                "tree_sha256": canonical_digest(mujoco_files),
            },
        },
        "validation": {"status": "pending"},
        "calibration": {"status": "pending"},
    }
    manifest_path = runtime / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    state_path.unlink()
    print(manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "finalize"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.repo.resolve(), args.source_archive.resolve() if args.source_archive else None)
    else:
        finalize(args.repo.resolve())


if __name__ == "__main__":
    main()
