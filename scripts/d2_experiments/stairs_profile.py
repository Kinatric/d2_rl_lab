"""Evaluate student on fixed stairs: python stairs_profile.py --policy P --output O."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
os.environ.setdefault('MUJOCO_GL','egl')
from _paths import HERE, ROOT
sys.path.insert(0,str(ROOT/'deploy/deploy_mujoco'))
import numpy as np
import mujoco
from deploy import VelocityDeployment,build_arg_parser
from d2_rl_lab.deployment import mujoco as d2
from utils import scene as scenery

parser=argparse.ArgumentParser()
parser.add_argument('--policy',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--height',type=float,choices=[.05,.10,.15],default=.10)
parser.add_argument('--delay',type=int,choices=range(5),default=2)
parser.add_argument('--seed',type=int,default=0)
parser.add_argument('--official',action='store_true')
parser.add_argument('--video',action='store_true')
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(args.seed);offset=float(rng.uniform(-.1,.1));yaw=float(rng.uniform(-math.pi/36,math.pi/36))
end_x=.8+12*.31+1+.8
original_scene=scenery.visual_scene_xml
def scene(path,terrain,urdf_name):
    root=ET.fromstring(original_scene(path,'stairs' if args.official else 'flat',urdf_name))
    if not args.official:
        world=root.find('worldbody')
        def box(name,start,width,height):
            if height>0:
                ET.SubElement(world,'geom',name=name,type='box',size=f'{width/2} .75 {height/2}',
                              pos=f'{start+width/2} 0 {height/2}',rgba='.65 .65 .65 1',friction='1 .005 .0001')
        for i in range(6):box(f'up_{i}',.8+i*.31,.31,(i+1)*args.height)
        box('platform',.8+6*.31,1,6*args.height)
        for i in range(6):box(f'down_{i}',.8+6*.31+1+i*.31,.31,(5-i)*args.height)
    return ET.tostring(root,encoding='unicode')
scenery.visual_scene_xml=scene
class FixedDelay(d2.PositionDelay):
    def reset(self):super().reset();self.lag=args.delay
d2.PositionDelay=FixedDelay
forward=mujoco.mj_forward;initialized=False
def initialize(model,data):
    global initialized
    if not initialized:
        data.qpos[0]=0;data.qpos[1]=offset;data.qpos[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
        initialized=True
    return forward(model,data)
mujoco.mj_forward=initialize
rows=[];steps=0;streak=0;failure=None;success=False;highest=0.;video=None;renderer=None
limit=120 if args.official else 60
class Done(Exception):pass
step=mujoco.mj_step
def measure(model,data,*a,**kw):
    global steps,streak,failure,success,highest,video,renderer
    step(model,data,*a,**kw);steps+=1
    finite=all(np.isfinite(v).all() for v in [data.qpos,data.qvel,data.ctrl])
    if not finite:failure='nonfinite';raise Done()
    qw,qx,qy,qz=data.qpos[3:7]
    tilt=math.degrees(math.acos(float(np.clip(1-2*(qx*qx+qy*qy),-1,1))))
    base_force=0.
    for i in range(data.ncon):
        contact=data.contact[i];ids=[int(model.geom_bodyid[g]) for g in [contact.geom1,contact.geom2]]
        if 0 in ids and any(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b)=='base' for b in ids):
            force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force);base_force+=abs(float(force[0]))
    streak=streak+1 if base_force>5 else 0
    warnings=int(sum(w.number for w in data.warning))
    if warnings:failure='mujoco_warning'
    elif tilt>45:failure='tilt'
    elif streak>=20:failure='base_contact_0.1s'
    elif abs(data.qpos[1])>.5:failure='lane_departure'
    for i in range(model.ngeom):
        if model.geom_bodyid[i]==0 and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX:
            top=model.geom_pos[i,2]+model.geom_size[i,2]
            if (data.qpos[0]>=model.geom_pos[i,0]+model.geom_size[i,0]
                and abs(data.qpos[1]-model.geom_pos[i,1])<model.geom_size[i,1]
                and data.qpos[2]>top+.1):highest=max(highest,float(top))
    if steps%4==0:
        rows.append([float(data.time),*data.qpos[:3],*data.qvel[:6],tilt,base_force,warnings])
        if args.video:
            if renderer is None:
                import imageio.v2 as imageio
                model.vis.global_.offwidth=960;model.vis.global_.offheight=540
                renderer=mujoco.Renderer(model,height=540,width=960)
                video=imageio.get_writer(args.output/'rollout.mp4',fps=50)
            cam=mujoco.MjvCamera();cam.distance=3.2;cam.azimuth=135;cam.elevation=-23;cam.lookat[:]=data.qpos[:3]+[.3,0,0]
            opt=mujoco.MjvOption();opt.geomgroup[3]=0
            renderer.update_scene(data,camera=cam,scene_option=opt);video.append_data(renderer.render())
    if not args.official and data.qpos[0]>=end_x:success=failure is None;raise Done()
    if failure or data.time>=limit-1e-9:raise Done()
mujoco.mj_step=measure
try:
    cli=build_arg_parser().parse_args(['--config',str(ROOT/'deploy/deploy_mujoco/d2.yaml'),
        '--policy',str(args.policy),'--terrain','stairs','--headless','--no-joystick','--no-real-time',
        '--duration','1800','--command',str(.5 if args.official else .3),'0','0'])
    try:VelocityDeployment(cli,cli.config).run()
    except Done:pass
finally:
    if video is not None:video.close()
    if renderer is not None:renderer.close()
result={'passed':success if not args.official else failure is None,'official':args.official,
        'height':args.height,'delay':args.delay,'seed':args.seed,'initial_y':offset,'initial_yaw':yaw,
        'failure':failure if failure else (None if success or args.official else 'timeout'),
        'physics_steps':steps,'seconds':steps*.005,'max_x':max((r[1] for r in rows),default=0),
        'highest_passed_surface_m':highest,'end_x':end_x,'continuous_history':True}
np.savetxt(args.output/'trajectory.csv',np.array(rows).reshape(-1,13),delimiter=',',
    header='time,x,y,z,vx,vy,vz,wx,wy,wz,tilt,base_force,warnings',comments='')
(args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
