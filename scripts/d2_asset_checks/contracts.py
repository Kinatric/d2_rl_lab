"""CPU checks for D2 torque, delay, mapping and checkpoint identity: python contracts.py."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
import mujoco

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tests/upgrade_isaaclab_3_ea/d2_asset_validation'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
from d2_rl_lab.assets.d2_contract import JOINT_NAMES, RL_POSE, RUNTIME, asset_manifest, motor_spec, policy_identity
from d2_rl_lab.actuators.robstride_actuator import RobStride06Actuator, RobStride06ActuatorCfg
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.math import quat_apply_inverse
from d2_rl_lab.deployment.mujoco import PositionDelay, clip_rs06, d2_observation
from utils.mujoco import get_joint_indices, get_gravity_orientation, quat_rotate_inverse
from rsl_rl.base.base_runner import BaseRunner

spec = motor_spec()
rng = np.random.default_rng(42)
max_error = 0.0
speed_nodes = np.array(spec["curve"])[:,0]
for lag in range(5):
    cfg = RobStride06ActuatorCfg(joint_names_expr=JOINT_NAMES, stiffness=25.0, damping=0.5, min_delay=lag, max_delay=lag)
    actuator = RobStride06Actuator(cfg, joint_names=JOINT_NAMES, joint_ids=slice(None), num_envs=1, device="cpu")
    delay = PositionDelay(lag, lag, seed=42)
    for episode in range(2):
        actuator.reset(None)
        delay.reset()
        for step in range(60):
            q = rng.uniform(-1, 1, (1,12)).astype("float32")
            target = rng.uniform(-2, 2, (1,12)).astype("float32")
            v = rng.choice(np.r_[speed_nodes, -speed_nodes, -60., 60.], (1,12)).astype("float32")
            action = ArticulationActions(torch.from_numpy(target.copy()), torch.zeros(1,12), torch.zeros(1,12))
            actual = actuator.compute(action, torch.from_numpy(q), torch.from_numpy(v)).joint_efforts.numpy()
            expected = clip_rs06(25*(delay.compute(target)-q)-0.5*v, v, spec)
            error = float(np.max(np.abs(actual-expected)))
            max_error = max(max_error,error)
            assert error < 1e-4, (lag, step, actual, expected)

frame = d2_observation(np.full(3,200.),np.full(3,-200.),np.full(3,200.),np.array(RL_POSE)+200.,np.full(12,-200.),np.full(12,200.))
assert np.array_equal(frame[:3],np.full(3,25.))
assert np.array_equal(frame[3:6],np.full(3,-100.))
assert np.array_equal(frame[6:9],np.array([50.,100.,100.]))
assert np.array_equal(frame[21:33],np.full(12,-5.))
assert np.array_equal(frame[33:],np.full(12,100.))

manifest = asset_manifest(False)
model = mujoco.MjModel.from_xml_path(str(RUNTIME/manifest["outputs"]["mujoco_scene"]["path"]))
data = mujoco.MjData(model)
qi, vi, ai = get_joint_indices(model, JOINT_NAMES)
feet = [mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,leg+"_foot") for leg in ("FL","FR","RL","RR")]
perm = np.array([3,4,5,0,1,2,9,10,11,6,7,8])
sign = np.array([-1,1,1]*4)
mirror_error = 0.0
for _ in range(30):
    q = np.array(RL_POSE)+rng.uniform(-0.1,0.1,12)
    data.qpos[qi] = q
    mujoco.mj_forward(model,data)
    before = data.xpos[feet].copy()
    data.qpos[qi] = q[perm]*sign
    mujoco.mj_forward(model,data)
    error = float(np.abs(data.xpos[feet]-before[[1,0,3,2]]*np.array([1,-1,1])).max())
    mirror_error = max(mirror_error,error)
assert mirror_error < 0.002, mirror_error

for _ in range(30):
    q = rng.normal(size=4).astype("float32"); q /= np.linalg.norm(q)
    v = rng.normal(size=3).astype("float32")
    reference = quat_apply_inverse(torch.tensor(q[[1,2,3,0]])[None], torch.tensor(v)[None]).numpy()[0]
    assert np.allclose(quat_rotate_inverse(q,v),reference,atol=1e-6)
    gravity = quat_apply_inverse(torch.tensor(q[[1,2,3,0]])[None], torch.tensor([[0.,0.,-1.]])).numpy()[0]
    assert np.allclose(get_gravity_orientation(q),gravity,atol=1e-6)

from d2_rl_lab.tasks.d2.mdp import base_ground_contact
# Newton base/thigh self-contact must not terminate; actual terrain contact must.
matrix = torch.zeros(2,3,17,1,3)
sensor = SimpleNamespace(data=SimpleNamespace(force_matrix_w_history=SimpleNamespace(torch=matrix)))
env = SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces":sensor}))
sensor_cfg = SimpleNamespace(name="contact_forces",body_ids=[0])
assert not base_ground_contact(env,sensor_cfg,1.0).any()
matrix[1,1,0,0,2] = 2.0
assert base_ground_contact(env,sensor_cfg,1.0).tolist() == [False,True]

identity = policy_identity(manifest=manifest)
test_checkpoint = OUT/'identity_fixture.pt'
dummy = SimpleNamespace(policy_identity=identity, _load_checkpoint_state=lambda data,load_optimizer: "accepted")
torch.save({"policy_identity":identity},test_checkpoint)
assert BaseRunner.load(dummy,str(test_checkpoint)) == "accepted"
negative_checks = []
for bad in [None, {**identity,"motor_sha256":"changed"}, {**identity,"history_length":1}]:
    torch.save({"policy_identity":bad},test_checkpoint)
    try: BaseRunner.load(dummy,str(test_checkpoint))
    except ValueError: negative_checks.append(True)
    else: raise AssertionError("Mismatched checkpoint accepted")
torch.save({"policy_identity":identity},test_checkpoint)
dummy.policy_identity = None
try: BaseRunner.load(dummy,str(test_checkpoint))
except ValueError: negative_checks.append(True)
else: raise AssertionError("D2 checkpoint accepted by an environment without its identity")
test_checkpoint.unlink()
report={"passed":True,"pd_delay_torque_cases":600,"max_torque_error_nm":max_error,
        "joint_mirror_error_m":mirror_error,"quaternion_cases":30,"rejected_checkpoint_cases":len(negative_checks)}
(OUT/'contracts.json').write_text(json.dumps(report,indent=2))
print("CONTRACTS_PASS",report)
