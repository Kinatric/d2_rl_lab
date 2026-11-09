"""Measure continuous walk/stop/walk through the production D2 controller.

Only the command input, prescribed delay and passive recording are instrumented.
Policy history, action delay and simulation are NOT reset at command boundaries.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
from _paths import ROOT
sys.path.insert(0, str(ROOT / 'deploy/deploy_mujoco'))
import mujoco
import numpy as np
from deploy import VelocityDeployment, build_arg_parser
from d2_rl_lab.deployment import mujoco as d2_utils
from utils.mujoco import quat_rotate_inverse

parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--profile', choices=['stand', 'forward', 'backward', 'lateral', 'turn'], required=True)
parser.add_argument('--delay', type=int, choices=range(5), default=2)
parser.add_argument('--move-command', nargs=3, type=float, help='Override moving phases for additional validation.')
parser.add_argument('--video', action='store_true')
parser.add_argument('--stand-seconds', type=float, default=30., help='Duration of the standalone zero-command test.')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
move = {'stand': [0, 0, 0], 'forward': [.5, 0, 0], 'backward': [-.5, 0, 0],
        'lateral': [0, .3, 0], 'turn': [0, 0, .5]}[args.profile]
if args.move_command is not None:
    assert args.profile != 'stand' and np.isfinite(args.move_command).all()
    move = args.move_command
assert args.stand_seconds > 2. and np.isfinite(args.stand_seconds)
phases = [(0, args.stand_seconds, [0, 0, 0])] if args.profile == 'stand' else [(0, 10, move), (10, 40, [0, 0, 0]), (40, 50, move)]
seconds = phases[-1][1]
records, command_calls = [], []
steps, warnings, fall_streak = 0, 0, 0
now, first_fall = 0., None
observed_command = np.array(move, dtype=float)
foot_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']


class Done(Exception):
    pass


class FixedDelay(d2_utils.PositionDelay):
    def reset(self):
        super().reset()
        self.lag = args.delay


d2_utils.PositionDelay = FixedDelay
original_observation = d2_utils.d2_observation


def observation(angular_velocity, gravity, command, *rest):
    global observed_command
    desired = next(c for start, end, c in phases if start <= now + 1e-9 < end)
    command[:] = desired
    observed_command = command.copy()
    result = original_observation(angular_velocity, gravity, command, *rest)
    assert np.allclose(result[6:9], np.array(desired) * [.5, 1, 1])
    command_calls.append([now, *observed_command])
    return result


d2_utils.d2_observation = observation
original_step = mujoco.mj_step


def measured_step(model, data, *positional, **keywords):
    global steps, now, warnings, fall_streak, first_fall
    original_step(model, data, *positional, **keywords)
    steps += 1
    assert float(data.time) > now, 'Unexpected simulation reset during profile'
    now = float(data.time)
    if not all(np.isfinite(v).all() for v in [data.qpos, data.qvel, data.ctrl]):
        raise FloatingPointError('Non-finite MuJoCo rollout')
    warnings = int(sum(w.number for w in data.warning))
    if steps % 4 == 0:
        qw, qx, qy, qz = data.qpos[3:7]
        upright = float(1 - 2 * (qx*qx + qy*qy))
        tilt = math.degrees(math.acos(float(np.clip(upright, -1, 1))))
        velocity = quat_rotate_inverse(data.qpos[3:7], data.qvel[:3])
        base_force, nonfoot_force = 0., 0.
        foot_forces = np.zeros(4)
        for index in range(data.ncon):
            contact = data.contact[index]
            bodies = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            if 0 not in bodies:
                continue
            body = bodies[0] if bodies[1] == 0 else bodies[1]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ''
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            load = abs(float(force[0]))
            if name in foot_names:
                foot_forces[foot_names.index(name)] += load
            if not name.endswith('_foot'):
                nonfoot_force += load
            if name == 'base':
                base_force += load
        fallen = data.qpos[2] < .13 or upright < .5 or base_force > 5.
        fall_streak = fall_streak + 1 if fallen else 0
        if fall_streak >= 5 and first_fall is None:
            first_fall = now
        records.append([now, *data.qpos[:3], *velocity, data.qvel[5], tilt, upright,
                        base_force, nonfoot_force, np.abs(data.ctrl).max(), *observed_command,
                        *foot_forces, np.sqrt(np.mean(np.square(data.qvel[6:])))])
    if now >= seconds - 1e-9:
        raise Done()


mujoco.mj_step = measured_step
if args.video:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    original_writer = imageio.get_writer
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()

    def annotated_writer(*positional, **kwargs):
        writer = original_writer(*positional, **kwargs)
        append = writer.append_data

        def annotated_frame(frame, *args, **kwargs):
            picture = Image.fromarray(frame)
            draw = ImageDraw.Draw(picture)
            draw.rectangle([0,0,picture.width,65],fill=(24,29,36))
            caption = f't={now:05.2f}s  command: vx={observed_command[0]:+.2f} vy={observed_command[1]:+.2f} yaw={observed_command[2]:+.2f}'
            draw.text((12,8),caption,font=font,fill=(245,246,248))
            if records:
                row=records[-1]
                draw.text((12,35),f'World x={row[1]:+.3f}m y={row[2]:+.3f}m   height={row[3]:.3f}m',font=font,fill=(245,246,248))
            return append(np.asarray(picture),*args,**kwargs)

        writer.append_data = annotated_frame
        return writer

    imageio.get_writer = annotated_writer
    original_renderer_init = mujoco.Renderer.__init__

    def bright_renderer(renderer, model, *positional, **kwargs):
        model.vis.headlight.ambient[:] = .65
        model.vis.headlight.diffuse[:] = .85
        model.vis.headlight.specular[:] = .1
        planes = np.where(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)[0]
        model.geom_size[planes, :2] = 100.
        model.geom_rgba[planes] = [.55, .58, .62, 1.]
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 960)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 540)
        kwargs.update(width=960, height=540)
        original_renderer_init(renderer, model, *positional, **kwargs)

    mujoco.Renderer.__init__ = bright_renderer
    original_update = mujoco.Renderer.update_scene
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 1.8, 125., -20.

    def render(renderer, data, **kwargs):
        camera.lookat[:] = data.qpos[:3]
        kwargs['camera'] = camera
        result = original_update(renderer, data, **kwargs)
        # World-anchored 25 cm grid, drawn only in the render scene. This makes
        # drift visible with a following camera without adding collision bodies.
        center = np.floor(data.qpos[:2]/.25)*.25
        scene = renderer.scene
        for offset in np.arange(-8,9)*.25:
            endpoints = [([center[0]+offset,center[1]-2.,.0015],[center[0]+offset,center[1]+2.,.0015]),
                         ([center[0]-2.,center[1]+offset,.0015],[center[0]+2.,center[1]+offset,.0015])]
            for start,end in endpoints:
                assert scene.ngeom < scene.maxgeom
                geom=scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),
                                   np.eye(3).flatten(),np.array([.25,.30,.35,.75],dtype=np.float32))
                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_LINE,1.25,np.array(start),np.array(end))
                scene.ngeom += 1
        return result

    mujoco.Renderer.update_scene = render

cli = ['--config', str(ROOT/'deploy/deploy_mujoco/d2.yaml'), '--policy', str(args.policy),
       '--headless', '--no-joystick', '--no-real-time', '--duration', '1200', '--command', *map(str, move)]
if args.video:
    cli += ['--save-video', '--video-path', str(args.output/'rollout.mp4')]
try:
    parsed = build_arg_parser().parse_args(cli)
    VelocityDeployment(parsed, parsed.config).run()
except Done:
    pass
assert steps == round(seconds/.005), f'Incomplete simulation: {steps} steps'
samples = np.array(records)
with (args.output/'trajectory.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time','x','y','z','vx','vy','vz','yaw_rate','tilt_deg','upright','base_force',
                     'nonfoot_force','peak_torque','cmd_x','cmd_y','cmd_yaw',*foot_names,'joint_speed_rms'])
    writer.writerows(samples)
np.savetxt(args.output/'command_inputs.csv', np.array(command_calls), delimiter=',',
           header='time,cmd_x,cmd_y,cmd_yaw', comments='')
results = []
for start, end, desired in phases:
    full = samples[(samples[:, 0] > start+1e-7) & (samples[:, 0] <= end+1e-7)]
    settled = full[full[:, 0] >= start+2.]
    error = settled[:, [4,5,7]] - desired
    rmse = np.sqrt(np.mean(error**2, axis=0))
    xy_rmse = float(np.sqrt(np.mean(np.sum(error[:, :2]**2, axis=1))))
    displacement = settled[:, 1:3] - settled[0, 1:3]
    max_displacement = float(np.linalg.norm(displacement, axis=1).max())
    mean_height = float(settled[:, 3].mean())
    max_tilt = float(settled[:, 8].max())
    nonfoot_fraction = float(np.mean(settled[:,11] > 5.))
    stable = first_fall is None and warnings == 0 and settled[:, 3].min() > .20 and max_tilt < 20. and nonfoot_fraction < .01
    is_stop = not any(desired)
    if is_stop:
        passed = stable and xy_rmse < .01 and rmse[2] < .05 and max_displacement < .10
        passed = passed and .285 <= mean_height <= .345 and max_tilt < 10.
    else:
        passed = stable and xy_rmse < .15 and rmse[2] < .25
        for k, target in enumerate(desired):
            if target:
                passed = passed and np.mean(settled[:, [4,5,7]][:, k]) * np.sign(target) >= .7*abs(target)
    quiet = (np.linalg.norm(full[:, 4:6], axis=1) < .02) & (np.abs(full[:, 7]) < .05)
    quiet_windows = np.flatnonzero(np.convolve(quiet.astype(int), np.ones(50, dtype=int), 'valid') == 50)
    results.append({'start_s':start,'end_s':end,'command':desired,'standing':is_stop,'passed':bool(passed),
        'settled_from_s':start+2.,'mean_velocity':settled[:, [4,5,7]].mean(axis=0).tolist(),
        'xy_rmse_m_s':xy_rmse,'rmse_vx_vy_yaw':rmse.tolist(),'mean_height_m':mean_height,
        'min_height_m':float(settled[:,3].min()),'max_tilt_deg':max_tilt,
        'nonfoot_contact_fraction':nonfoot_fraction,
        'settled_net_displacement_xy_m':displacement[-1].tolist(),
        'settled_max_displacement_m':max_displacement,
        'first_2s_displacement_m':float(np.linalg.norm(settled[0,1:3]-full[0,1:3])),
        'quiet_onset_s':float(full[quiet_windows[0],0]-start) if is_stop and len(quiet_windows) else None,
        'foot_load_fraction':np.mean(settled[:,16:20]>5.,axis=0).tolist(),
        'foot_contact_transitions':np.sum(np.diff(settled[:,16:20]>5.,axis=0)!=0,axis=0).tolist(),
        'joint_speed_rms_rad_s':float(np.sqrt(np.mean(settled[:,-1]**2)))})
result = {'policy':str(args.policy),'profile':args.profile,'delay_steps':args.delay,'seconds':seconds,
          'physics_steps':steps,'warnings':warnings,'first_fall_s':first_fall,'finite':True,
          'continuous_history':True,'phases':results,'passed':all(p['passed'] for p in results)}
(args.output/'result.json').write_text(json.dumps(result, indent=2))
print('\nPROFILE_RESULT',json.dumps(result),flush=True)
