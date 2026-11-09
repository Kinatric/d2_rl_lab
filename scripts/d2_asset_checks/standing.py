"""Validate physical assets and 30 s standing: python standing.py --backend mujoco|newton_mjwarp|isaacsim_physx --viz none."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tests/upgrade_isaaclab_3_ea/d2_asset_validation'
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser()
parser.add_argument("--backend", required=True)
add_launcher_args(parser)
args = parser.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
from d2_rl_lab.assets.d2_contract import (
    DESCRIPTION, RUNTIME, JOINT_NAMES, RL_POSE, FIXSTAND_POSE,
    asset_manifest, runtime_identity, motor_spec, pose_height,
)
manifest = asset_manifest(False)
report = {"backend": args.backend, "runtime_identity": runtime_identity(manifest)}
heights, forces, orientations, speeds, nonfoot = [], [], [], [], []
from static_assets import validate_urdf, validate_usd, validate_mujoco
urdf = validate_urdf(DESCRIPTION / "urdf/d2_description.urdf")
report["urdf"] = urdf

if args.backend == "mujoco":
    import mujoco
    from d2_rl_lab.deployment.mujoco import PositionDelay, clip_rs06
    from utils.mujoco import get_joint_indices
    scene = RUNTIME / manifest["outputs"]["mujoco_scene"]["path"]
    report["static"] = validate_mujoco(scene, urdf, 0)
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = 0.005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 100
    data = mujoco.MjData(model)
    qi, vi, ai = get_joint_indices(model, JOINT_NAMES)
    assert np.allclose(model.dof_armature[vi], 0.012)
    assert np.allclose(model.dof_frictionloss[vi], 0.01)
    data.qpos[2], data.qpos[qi] = pose_height(FIXSTAND_POSE)+0.03, RL_POSE
    delay, motor = PositionDelay(seed=42), motor_spec()
    foot_seen = set()
    target = np.asarray(RL_POSE)
    for step in range(6000):
        torque = clip_rs06(25*(delay.compute(target)-data.qpos[qi])-0.5*data.qvel[vi], data.qvel[vi], motor)
        data.ctrl[ai] = torque
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert np.max(np.abs(torque)) <= 36.0001
        if step >= 4000:
            heights.append(float(data.qpos[2]))
            orientations.append(float(1-2*(data.qpos[4]**2+data.qpos[5]**2)))
            speeds.append(float(np.abs(data.qvel[vi]).max()))
            total, other = 0.0, 0.0
            for i in range(data.ncon):
                contact = data.contact[i]
                body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) for g in (contact.geom1, contact.geom2)]
                cf = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, cf)
                if "world" in body_names:
                    total += cf[0]
                    feet = [n for n in body_names if n.endswith("_foot")]
                    foot_seen.update(feet)
                    if not feet:
                        other += cf[0]
            forces.append(total)
            nonfoot.append(other)
    report["foot_contacts_seen"] = sorted(foot_seen)
    assert len(foot_seen) == 4
    assert not any(w.number for w in data.warning), "MuJoCo warnings"
else:
    import runtime_args
    args.physics, args.video = args.backend, False
    runtime_args.bootstrap_runtime(args)
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.utils.hydra import resolve_presets
    from d2_rl_lab.tasks.d2.flat_env_cfg import FlatVelocityRobotEnvCfg
    from d2_rl_lab.assets.d2 import make_d2_cfg
    # A private validation config avoids task registration and readiness recursion:
    # standing is itself one of the checks required to mark new assets ready.
    cfg = resolve_presets(FlatVelocityRobotEnvCfg(), selected=(args.backend,))
    cfg.scene.num_envs = 2
    cfg.seed = 42
    cfg.scene.robot = make_d2_cfg(require_ready=False)
    if args.backend == "newton_mjwarp":
        cfg.scene.contact_forces.filter_shape_prim_expr = [cfg.scene.terrain.prim_path+"/.*"]

    reset_joints, reset_base = cfg.events.reset_robot_joints, cfg.events.reset_base
    for group in (cfg.events, cfg.terminations, cfg.curriculum):
        for name in list(vars(group)):
            if not name.startswith("_"):
                setattr(group, name, None)
    reset_joints.params["position_range"] = (1.0, 1.0)
    reset_joints.params["velocity_range"] = (0.0, 0.0)
    reset_base.params["pose_range"] = {key: (0.0, 0.0) for key in reset_base.params["pose_range"]}
    reset_base.params["velocity_range"] = {key: (0.0, 0.0) for key in reset_base.params["velocity_range"]}
    cfg.events.reset_robot_joints, cfg.events.reset_base = reset_joints, reset_base
    cfg.episode_length_s = 100.0
    cfg.observations.policy.enable_corruption = False
    print("INIT_CONFIG", cfg.scene.robot.init_state, flush=True)
    report["static"] = validate_usd(RUNTIME / manifest["outputs"]["isaac_usd"]["path"], urdf)
    launch_args = vars(args).copy()
    launch_args["physics"] = None
    with launch_simulation(cfg, launch_args):
        env = ManagerBasedRLEnv(cfg=cfg)
        try:
            env.reset()
            robot = env.scene["robot"]
            sensor = env.scene["contact_forces"]
            foot_ids = [i for i,n in enumerate(sensor.body_names) if n.endswith("_foot")]
            other_ids = [i for i,n in enumerate(sensor.body_names) if not n.endswith("_foot")]
            assert len(foot_ids) == 4
            print("INITIAL_STATE", robot.data.root_state_w.torch[0].tolist(), "JOINTS", robot.joint_names, robot.data.joint_pos.torch[0].tolist(), flush=True)
            for step in range(1500):
                env.step(torch.zeros((2, 12), device=env.device))
                assert torch.isfinite(robot.data.joint_pos.torch).all()
                assert torch.isfinite(robot.data.root_pos_w.torch).all()
                if step in (0, 10, 50, 100):
                    print("TRACE", step, robot.data.root_state_w.torch[0].tolist(), robot.data.joint_pos.torch[0].tolist(), flush=True)
                if step >= 1000:
                    heights.append(float(robot.data.root_pos_w.torch[0,2]))
                    orientations.append(float(-robot.data.projected_gravity_b.torch[0,2]))
                    speeds.append(float(robot.data.joint_vel.torch[0].abs().max()))
                    force = (sensor.data.force_matrix_w.torch[0].sum(dim=1) if args.backend == "newton_mjwarp"
                             else sensor.data.net_forces_w.torch[0])
                    forces.append(float(force[foot_ids,2].sum()))
                    nonfoot.append(float(force[other_ids].norm(dim=-1).max()))
            report["settled_all_contact_forces"] = dict(zip(sensor.body_names, sensor.data.net_forces_w.torch[0].tolist()))
            report["settled_ground_contact_forces"] = dict(zip(sensor.body_names, force.tolist()))
            report["settled_joints"] = dict(zip(robot.joint_names, robot.data.joint_pos.torch[0].tolist()))
            before = robot.data.joint_pos.torch[1].clone()
            env._reset_idx(torch.tensor([0], device=env.device))
            assert torch.equal(before, robot.data.joint_pos.torch[1])
            report["partial_reset_preserved_other_environment"] = True
        finally:
            env.close()

report.update(duration_s=30, finite=True, median_height_m=float(np.median(heights)), height_span_m=float(np.ptp(heights)),
              median_support_n=float(np.median(forces)), min_upright=float(min(orientations)),
              max_settled_joint_speed=float(max(speeds)), max_nonfoot_contact_n=float(max(nonfoot)))
(OUT / (args.backend+"_standing.json")).write_text(json.dumps(report, indent=2))
assert 0.15 < report["median_height_m"] < 0.36, report
assert report["height_span_m"] < 0.015, report
assert report["min_upright"] > 0.9, report
assert abs(report["median_support_n"]-16.1086*9.81) < 0.1*16.1086*9.81, report
assert report["max_nonfoot_contact_n"] < 1.0, report
report["status"] = "passed"
(OUT / (args.backend+"_standing.json")).write_text(json.dumps(report, indent=2))
print("STANDING_PASS", args.backend, {k:v for k,v in report.items() if k not in ("static", "urdf", "settled_all_contact_forces", "settled_ground_contact_forces", "settled_joints")})

if args.backend != "mujoco":
    runtime_args.shutdown_runtime()
