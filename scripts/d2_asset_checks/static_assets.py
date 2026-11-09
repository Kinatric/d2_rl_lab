# Adapted from lookc4/go2_rl_robotlab 079b316; Apache-2.0. Static validation only.

import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

JOINT_ORDER = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]

FIXED_FOOT_JOINTS = [f"{leg}_foot_joint" for leg in ("FL", "FR", "RL", "RR")]

EXPECTED_LINK_MASSES = {
    "base": 5.425,
    **{f"{leg}_hip": 0.7737 for leg in ("FL", "FR", "RL", "RR")},
    **{f"{leg}_thigh": 1.6266 for leg in ("FL", "FR", "RL", "RR")},
    **{f"{leg}_calf": 0.2013 for leg in ("FL", "FR", "RL", "RR")},
    **{f"{leg}_foot": 0.0693 for leg in ("FL", "FR", "RL", "RR")},
}

EXPECTED_TOTAL_MASS_KG = 16.1086

EXPECTED_MASS_METHOD = "scale_full_inertia_tensor_by_new_mass_over_old_mass_preserving_com_geometry_and_inertial_frame"

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

def rotation_from_wxyz(quaternion: list[float] | np.ndarray) -> np.ndarray:
    w, x, y, z = map(float, quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0:
        raise RuntimeError(f"Invalid zero quaternion: {quaternion}")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])

def validate_urdf(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    links = root.findall("link")
    joints = root.findall("joint")
    revolute = [joint for joint in joints if joint.get("type") == "revolute"]
    fixed = [joint for joint in joints if joint.get("type") == "fixed"]
    if [joint.get("name") for joint in revolute] != JOINT_ORDER:
        raise RuntimeError("URDF revolute joint order differs.")
    if [joint.get("name") for joint in fixed] != FIXED_FOOT_JOINTS:
        raise RuntimeError("URDF fixed foot joints differ.")
    total_mass = 0.0
    link_properties = {}
    limits = {}
    for link in links:
        inertial = link.find("inertial")
        if inertial is None:
            raise RuntimeError(f"URDF link has no inertial: {link.get('name')}")
        mass = float(inertial.find("mass").get("value"))
        tensor = inertial.find("inertia")
        inertia = np.asarray([
            [float(tensor.get("ixx")), float(tensor.get("ixy")), float(tensor.get("ixz"))],
            [float(tensor.get("ixy")), float(tensor.get("iyy")), float(tensor.get("iyz"))],
            [float(tensor.get("ixz")), float(tensor.get("iyz")), float(tensor.get("izz"))],
        ])
        if not np.isfinite(inertia).all() or not math.isfinite(mass) or mass <= 0:
            raise RuntimeError(f"URDF mass/inertia is not positive for {link.get('name')}.")
        eigenvalues = np.linalg.eigvalsh(inertia)
        if np.min(eigenvalues) <= 0 or eigenvalues[0] + eigenvalues[1] < eigenvalues[2] - 1.0e-12:
            raise RuntimeError(f"URDF inertia is not physically valid for {link.get('name')}: {eigenvalues}")
        name = str(link.get("name"))
        expected_mass = EXPECTED_LINK_MASSES.get(name)
        if expected_mass is None or not math.isclose(mass, expected_mass, abs_tol=1.0e-12):
            raise RuntimeError(f"URDF measured mass differs for {name}: expected={expected_mass}, actual={mass}")
        origin = inertial.find("origin")
        com = [float(value) for value in origin.get("xyz", "0 0 0").split()]
        inertial_rpy = [float(value) for value in origin.get("rpy", "0 0 0").split()]
        rotation = rotation_from_rpy(inertial_rpy)
        link_inertia = rotation @ inertia @ rotation.T
        link_properties[name] = {
            "mass_kg": mass, "com_m": com, "inertial_frame_rpy_rad": inertial_rpy,
            "inertia_kg_m2": inertia.tolist(), "link_frame_inertia_kg_m2": link_inertia.tolist(),
        }
        total_mass += mass
    for joint in revolute:
        limit = joint.find("limit")
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise RuntimeError(f"Invalid joint limit for {joint.get('name')}.")
        limits[joint.get("name")] = [lower, upper]
    if not math.isclose(total_mass, EXPECTED_TOTAL_MASS_KG, abs_tol=1.0e-12):
        raise RuntimeError(f"URDF total mass differs: expected={EXPECTED_TOTAL_MASS_KG}, actual={total_mass}")
    return {
        "link_count": len(links), "collision_count": len(root.findall('.//collision')),
        "total_mass_kg": total_mass, "nominal_weight_n": total_mass * 9.81,
        "static_weight_per_foot_n": total_mass * 9.81 / 4.0,
        "joint_limits_rad": limits, "link_properties": link_properties,
    }

def validate_usd(path: Path, urdf_summary: dict[str, object]) -> dict[str, object]:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {path}")
    names = {prim.GetName() for prim in stage.Traverse()}
    missing_feet = [f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR") if f"{leg}_foot" not in names]
    missing_joints = [name for name in JOINT_ORDER if name not in names]
    if missing_feet or missing_joints:
        raise RuntimeError(f"USD lost fixed feet or joints: feet={missing_feet}, joints={missing_joints}")
    prims = {prim.GetName(): prim for prim in stage.Traverse() if prim.GetName() in EXPECTED_LINK_MASSES}
    if set(prims) != set(EXPECTED_LINK_MASSES):
        raise RuntimeError(f"USD link inventory differs: {set(prims) ^ set(EXPECTED_LINK_MASSES)}")
    usd_masses = {}
    usd_coms = {}
    for name, expected in urdf_summary["link_properties"].items():
        mass_api = UsdPhysics.MassAPI(prims[name])
        actual = mass_api.GetMassAttr().Get()
        if actual is None or not math.isclose(float(actual), float(expected["mass_kg"]), abs_tol=1.0e-6):
            raise RuntimeError(f"USD mass differs for {name}: expected={expected['mass_kg']}, actual={actual}")
        com = mass_api.GetCenterOfMassAttr().Get()
        diagonal = mass_api.GetDiagonalInertiaAttr().Get()
        axes = mass_api.GetPrincipalAxesAttr().Get()
        if com is None or diagonal is None or axes is None:
            raise RuntimeError(f"USD mass properties are incomplete for {name}.")
        actual_com = np.asarray(tuple(com), dtype=np.float64)
        imaginary = axes.GetImaginary()
        rotation = rotation_from_wxyz([axes.GetReal(), imaginary[0], imaginary[1], imaginary[2]])
        actual_inertia = rotation @ np.diag(np.asarray(tuple(diagonal), dtype=np.float64)) @ rotation.T
        expected_com = np.asarray(expected["com_m"], dtype=np.float64)
        expected_inertia = np.asarray(expected["link_frame_inertia_kg_m2"], dtype=np.float64)
        if not np.allclose(actual_com, expected_com, atol=1.0e-6, rtol=0):
            raise RuntimeError(f"USD center of mass differs for {name}: expected={expected_com}, actual={actual_com}")
        if not np.allclose(actual_inertia, expected_inertia, atol=1.0e-8, rtol=1.0e-4):
            raise RuntimeError(f"USD inertia differs for {name}: expected={expected_inertia}, actual={actual_inertia}")
        usd_masses[name] = float(actual)
        usd_coms[name] = actual_com.tolist()
    usd_limits = {}
    for name, expected in urdf_summary["joint_limits_rad"].items():
        joint_prims = [prim for prim in stage.Traverse() if prim.GetName() == name]
        if len(joint_prims) != 1:
            raise RuntimeError(f"USD joint prim differs for {name}: {len(joint_prims)} matches")
        joint = UsdPhysics.RevoluteJoint(joint_prims[0])
        actual = np.deg2rad([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()])
        if not np.allclose(actual, expected, atol=1.0e-5, rtol=0):
            raise RuntimeError(f"USD joint limits differ for {name}: expected={expected}, actual={actual}")
        usd_limits[name] = actual.tolist()
    # Collision shapes live below instanceable collision Xforms, so include
    # instance proxies rather than counting only ordinary composed prims.
    collision_prims = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
    collision_count = sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in collision_prims)
    if collision_count != 25:
        raise RuntimeError(f"USD collision count differs: expected=25, actual={collision_count}")
    return {
        "prim_count": len(names), "fixed_feet_preserved": True, "joint_order_preserved": True,
        "link_mass_com_inertia_parity": True, "link_masses_kg": usd_masses, "link_coms_m": usd_coms,
        "joint_limit_parity": True, "joint_limits_rad": usd_limits, "collision_count": collision_count,
    }

def validate_mujoco(path: Path, urdf_summary: dict[str, object], duration_s: float) -> dict[str, object]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)]
    hinge_names = [name for index, name in enumerate(names) if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE]
    if hinge_names != JOINT_ORDER:
        raise RuntimeError(f"MuJoCo joint order differs: {hinge_names}")
    ranges = {name: list(map(float, model.jnt_range[index])) for index, name in enumerate(names) if name in JOINT_ORDER}
    for name, expected in urdf_summary["joint_limits_rad"].items():
        if not np.allclose(ranges[name], expected, atol=1e-9, rtol=0):
            raise RuntimeError(f"MuJoCo joint limits differ for {name}: {ranges[name]} != {expected}")
    if not np.isclose(float(np.sum(model.body_mass)), urdf_summary["total_mass_kg"], atol=1e-6, rtol=0):
        raise RuntimeError("MuJoCo total mass differs from URDF.")
    body_masses = {}
    body_coms = {}
    for name, expected in urdf_summary["link_properties"].items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise RuntimeError(f"MuJoCo body is missing: {name}")
        actual = float(model.body_mass[body_id])
        if not math.isclose(actual, float(expected["mass_kg"]), abs_tol=1.0e-6):
            raise RuntimeError(f"MuJoCo mass differs for {name}: expected={expected['mass_kg']}, actual={actual}")
        actual_com = np.asarray(model.body_ipos[body_id], dtype=np.float64)
        actual_rotation = rotation_from_wxyz(model.body_iquat[body_id])
        actual_inertia = actual_rotation @ np.diag(np.asarray(model.body_inertia[body_id], dtype=np.float64)) @ actual_rotation.T
        expected_com = np.asarray(expected["com_m"], dtype=np.float64)
        expected_inertia = np.asarray(expected["link_frame_inertia_kg_m2"], dtype=np.float64)
        # The MJCF writer serializes these values with fewer decimal places than
        # the source URDF, so compare within its sub-micrometre text precision.
        if not np.allclose(actual_com, expected_com, atol=1.0e-7, rtol=0):
            raise RuntimeError(f"MuJoCo center of mass differs for {name}: expected={expected_com}, actual={actual_com}")
        # MuJoCo diagonalizes fullinertia internally and exposes the result as
        # principal moments plus a quaternion; reconstruction has small
        # eigendecomposition round-off, especially in off-diagonal terms.
        if not np.allclose(actual_inertia, expected_inertia, atol=1.0e-8, rtol=1.0e-4):
            raise RuntimeError(f"MuJoCo inertia differs for {name}: expected={expected_inertia}, actual={actual_inertia}")
        body_masses[name] = actual
        body_coms[name] = actual_com.tolist()
    if int(model.ngeom) - 1 != int(urdf_summary["collision_count"]):
        raise RuntimeError(f"MuJoCo collision count differs: expected={urdf_summary['collision_count']}, actual={int(model.ngeom) - 1}")

    return {"link_mass_com_inertia_parity": True, "joint_limit_parity": True, "mass_kg": float(model.body_mass.sum())}
