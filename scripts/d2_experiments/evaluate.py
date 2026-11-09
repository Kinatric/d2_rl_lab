"""One-shot serial export, screening, full acceptance and evidence generation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import gzip
import hashlib
from common import HERE, ROOT, atomic_json, sha, finite, TARGET_ITERATIONS, retain_checkpoint
from score import score


SCREEN=[('stand_d2','stand',2,[0,0,0]),('forward_d2','forward',2,[.5,0,0])]
FULL=[(f'stand_d{d}','stand',d,[0,0,0]) for d in [0,2,4]]
for label,profile,command in [('forward','forward',[.5,0,0]),('backward','backward',[-.5,0,0]),
    ('left','lateral',[0,.3,0]),('right','lateral',[0,-.3,0]),
    ('turn_left','turn',[0,0,.5]),('turn_right','turn',[0,0,-.5])]:
    FULL.extend((f'{label}_d{d}',profile,d,command) for d in [0,2,4])
FULL.extend((f'forward_{speed}_d{d}','forward',d,[speed,0,0]) for speed in [.3,.7] for d in [1,3])


def trajectory_path(folder):
    file=Path(folder)/'trajectory.csv'
    return file if file.exists() else file.with_suffix('.csv.gz')


def compress_csv(folder):
    """Losslessly compress only this experiment's generated CSVs, verifying every byte."""
    folder=Path(folder).resolve();assert folder.is_relative_to(HERE.resolve())
    for source in folder.glob('*.csv'):
        assert source.resolve().is_relative_to(HERE.resolve())
        target=source.with_suffix('.csv.gz');temporary=target.with_suffix('.gz.tmp')
        with source.open('rb') as raw,gzip.open(temporary,'wb',compresslevel=6) as compressed:
            shutil.copyfileobj(raw,compressed)
        digest=hashlib.sha256()
        with gzip.open(temporary,'rb') as compressed:
            while chunk:=compressed.read(1024*1024):digest.update(chunk)
        assert digest.hexdigest()==sha(source),'Compressed trajectory integrity failure'
        temporary.replace(target);source.unlink()


def run(command,log,timeout=900):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
    with Path(log).open('w') as file:
        result=subprocess.run(list(map(str,command)),cwd=ROOT,env=env,stdout=file,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:raise RuntimeError(f'Exit {result.returncode}: {log}')


def check_exports(folder):
    import numpy as np
    import torch
    import onnx
    torch.set_num_threads(1);torch.manual_seed(42)
    folder=Path(folder);jit=torch.jit.load(str(folder/'policy.pt'),map_location='cpu').eval();jit.reset()
    onnx.checker.check_model(onnx.load(str(folder/'policy.onnx')))
    try:
        import onnxruntime as ort
    except ImportError:
        from onnx.reference import ReferenceEvaluator
        session=ReferenceEvaluator(str(folder/'policy.onnx'));input_name=session.input_names[0]
        backend='onnx.reference'
    else:
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(folder/'policy.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        input_name=session.get_inputs()[0].name;backend='onnxruntime.CPU'
    frames=[];largest=0.
    for i in range(40):
        frame=torch.randn(1,45)*.2
        frame[:,6:9]=torch.tensor([[.25,0,0]]) if i<10 or i>=30 else 0.
        frames.append(frame)
        with torch.inference_mode():action=jit(frame)
        finite(action);assert action.shape==(1,12)
        history=torch.cat([torch.zeros(1,45)]*max(0,10-len(frames))+frames[-10:],dim=0)
        expected=torch.cat([history[:,a:b].reshape(-1) for a,b in [(0,3),(3,6),(6,9),(9,21),(21,33),(33,45)]])[None,:]
        assert torch.equal(jit.obs_history,expected),'History layout mismatch'
        result=session.run(None,{input_name:expected.numpy()})[0]
        assert np.isfinite(result).all() and np.allclose(result,action.numpy(),rtol=1e-5,atol=1e-5)
        largest=max(largest,float(np.abs(result-action.numpy()).max()))
    jit.reset();assert torch.count_nonzero(jit.obs_history)==0
    durations=[]
    with torch.inference_mode():
        for i in range(220):
            start=time.perf_counter();jit(frame);elapsed=time.perf_counter()-start
            if i>=20:durations.append(elapsed)
    latency={'median_ms':float(np.median(durations)*1000),'p95_ms':float(np.percentile(durations,95)*1000),'within_20ms':bool(np.percentile(durations,95)<.02)}
    record={'latency':latency,'finite':True,'frames':40,'torchscript_single_observation':45,'onnx_history_observation':450,
        'history_layout_and_reset_verified':True,'max_jit_onnx_difference':largest,
        'onnx_inference_backend':backend,
        'files':{f.name:sha(f) for f in folder.iterdir() if f.is_file() and f.name!='inference_check.json'}}
    atomic_json(folder/'inference_check.json',record)
    return record


def export_checkpoint(source,folder):
    folder.mkdir(parents=True,exist_ok=True);checkpoint=folder/source.name
    retain_checkpoint(source,checkpoint)
    assert sha(source)==sha(checkpoint)
    if not (folder/'params').exists():shutil.copytree(source.parent/'params',folder/'params')
    marker=folder/'export_complete.json'
    if marker.exists():
        saved=json.loads(marker.read_text());assert saved['checkpoint_sha256']==sha(checkpoint)
        return folder/'exported/policy.pt'
    run(['bash',ROOT/'d2_rl_lab.sh','--play','--task','d2-mixed-moe-cts','--physics','newton_mjwarp',
        '--viz','none','--num_envs','4','--max_steps','100','--checkpoint',checkpoint],folder/'export.log')
    checked=check_exports(folder/'exported')
    atomic_json(marker,{'checkpoint_sha256':sha(checkpoint),'exports':checked})
    return folder/'exported/policy.pt'


def evaluate_case(policy,base,spec,video=False):
    name,profile,delay,command=spec;folder=base/(name+('_video' if video else ''));folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'case_complete.json';identity={'policy_sha256':sha(policy),'spec':list(spec),'video':video,'score_sha256':sha(HERE/'score.py')}
    if marker.exists():
        result=json.loads(marker.read_text())
        assert result['identity']==identity
        if 'error' not in result:return result
        marker.rename(folder/f'case_failed_{time.time_ns()}.json')
        if (folder/'run.log').exists():shutil.copy2(folder/'run.log',folder/f'run_failed_{time.time_ns()}.log')
    try:
        arguments=[os.sys.executable,HERE/'mujoco_profile.py','--policy',policy,'--output',folder,
            '--profile',profile,'--delay',delay,'--stand-seconds',120]
        if profile!='stand':arguments+=['--move-command',*command]
        if video:arguments+=['--video']
        run(arguments,folder/'run.log')
        result=score(folder);result.update(identity=identity,folder=str(folder))
        if video:
            import imageio.v2 as imageio
            import numpy as np
            reader=imageio.get_reader(folder/'rollout.mp4');count=reader.count_frames()
            assert count==round(result['seconds']*50),(count,result['seconds'])
            snapshots=[]
            for i in [0,count//2,count-1]:
                frame=reader.get_data(i);assert frame.std()>10
                imageio.imwrite(folder/f'frame_{i}.png',frame);snapshots.append(str(folder/f'frame_{i}.png'))
            reader.close()
            reference=np.loadtxt(trajectory_path(base/name),delimiter=',',skiprows=1)
            rendered=np.loadtxt(trajectory_path(folder),delimiter=',',skiprows=1)
            assert reference.shape==rendered.shape
            difference=float(np.max(np.abs(reference-rendered)))
            result['video_check']={'frames':count,'trajectory_max_abs_difference':difference,'snapshots':snapshots}
        compress_csv(folder)
    except Exception:
        result={'passed':False,'identity':identity,'folder':str(folder),'error':traceback.format_exc()}
    atomic_json(marker,result)
    return result


def stair_case(policy,base,height=.1,delay=2,seed=0,official=False,video=False):
    name=f"{'official' if official else str(height)}_d{delay}_s{seed}"+('_video' if video else '')
    folder=base/name;folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'case_complete.json'
    identity={'policy':sha(policy),'script':sha(HERE/'stairs_profile.py'),'height':height,'delay':delay,'seed':seed,'official':official,'video':video}
    if marker.exists():
        result=json.loads(marker.read_text())
        assert result['identity']==identity
        return result
    try:
        cmd=[os.sys.executable,HERE/'stairs_profile.py','--policy',policy,'--output',folder,'--height',height,'--delay',delay,'--seed',seed]
        if official:cmd+=['--official']
        if video:cmd+=['--video']
        run(cmd,folder/'run.log',timeout=1800)
        result=json.loads((folder/'result.json').read_text());result['folder']=str(folder)
        if video:
            import imageio.v2 as imageio
            reader=imageio.get_reader(folder/'rollout.mp4');count=reader.count_frames()
            assert count>0
            for i in [0,count//2,count-1]:imageio.imwrite(folder/f'frame_{i}.png',reader.get_data(i))
            reader.close();result['video_frames']=count
        compress_csv(folder)
    except Exception:result={'passed':False,'error':traceback.format_exc()}
    result['identity']=identity;atomic_json(marker,result);return result


def main(checkpoint):
    checkpoint=Path(checkpoint)
    state={'stage':'screening','candidates':{},'baselines':{},'recommended_iteration':None}
    def persist():atomic_json(HERE/'evaluation_status.json',state)
    for iteration in [*range(500,TARGET_ITERATIONS,500),TARGET_ITERATIONS-1]:
        label=f'model_{iteration}';folder=HERE/'evaluation'/label
        item={'iteration':iteration,'cases':{},'stairs':{},'full_acceptance':False}
        state['candidates'][label]=item;state['current']=label;persist()
        try:
            policy=export_checkpoint(checkpoint.parent/f'{label}.pt',folder);item['policy']=str(policy)
            item['inference']=json.loads((folder/'exported/inference_check.json').read_text())
            for spec in SCREEN:item['cases'][spec[0]]=evaluate_case(policy,folder,spec);persist()
            trials=[stair_case(policy,folder/'stairs',seed=i) for i in range(3)]
            item['stairs_screen']=trials;item['stairs_rate']=sum(t['passed'] for t in trials)/3
            item['screen_passed']=all(c['passed'] for c in item['cases'].values())
            item['speed_error']=max(c.get('worst_relative_speed_error',1e9) for c in item['cases'].values())
            item['drift']=max(c.get('max_stand_drift_m',1e9) for c in item['cases'].values())
        except Exception:item['error']=traceback.format_exc()
        persist()
    eligible=[c for c in state['candidates'].values() if 'error' not in c]
    shortlist=sorted(eligible,key=lambda c:(-c['stairs_rate'],-int(c['screen_passed']),c['speed_error'],c['drift'],-c['iteration']))[:5]
    ids={c['iteration'] for c in shortlist}|{TARGET_ITERATIONS-1}
    state['shortlist']=sorted(ids);state['stage']='full_acceptance';persist()
    for iteration in sorted(ids):
        item=state['candidates'][f'model_{iteration}']
        if 'policy' not in item:continue
        policy=Path(item['policy']);folder=policy.parents[1]
        try:
            for spec in FULL:
                if spec[0] not in item['cases']:item['cases'][spec[0]]=evaluate_case(policy,folder,spec);persist()
            for height in [.05,.10,.15]:
                for delay in [0,2,4]:
                    key=f'{height}_d{delay}'
                    trials=[stair_case(policy,folder/'stairs',height,delay,i) for i in range(10)]
                    item['stairs'][key]={'trials':trials,'success_rate':sum(t['passed'] for t in trials)/10};persist()
            item['official']=[stair_case(policy,folder/'stairs',delay=d,official=True) for d in [0,2,4]]
            item['worst_stairs_rate']=min(c['success_rate'] for c in item['stairs'].values())
            item['speed_error']=max(c.get('worst_relative_speed_error',1e9) for c in item['cases'].values())
            item['drift']=max(c.get('max_stand_drift_m',1e9) for c in item['cases'].values())
            item['full_acceptance']=(all(item['cases'][spec[0]]['passed'] for spec in FULL)
                and item['worst_stairs_rate']>=.9 and item['inference']['latency']['within_20ms'])
        except Exception:item['full_error']=traceback.format_exc()
        persist()
    passing=[c for c in state['candidates'].values() if c['full_acceptance']]
    if passing:
        best=min(passing,key=lambda c:(-c['worst_stairs_rate'],c['speed_error'],c['drift'],-c['iteration']))
        state['recommended_iteration']=best['iteration']
    baseline_path=json.loads((HERE/'experiment.json').read_text()).get('baseline_policy')
    if baseline_path:
        baseline=Path(baseline_path).expanduser().resolve()
        state['stage']='baseline_comparison';persist()
        base=HERE/'comparison/baseline'
        state['baselines']['baseline']={'flat':{s[0]:evaluate_case(baseline,base,s) for s in FULL},
            'stairs':{f'{h}_d{d}':[stair_case(baseline,base/'stairs',h,d,i) for i in range(10)] for h in [.05,.1,.15] for d in [0,2,4]},
            'official':[stair_case(baseline,base/'stairs',delay=d,official=True) for d in [0,2,4]]}
    state['stage']='videos';persist()
    for iteration in sorted({TARGET_ITERATIONS-1,state['recommended_iteration']}-{None}):
        item=state['candidates'][f'model_{iteration}']
        if 'policy' not in item:continue
        policy=Path(item['policy']);folder=policy.parents[1]
        item['videos']={s[0]:evaluate_case(policy,folder,s,True) for s in SCREEN}
        item['videos']['stairs']=stair_case(policy,folder/'stairs',video=True)
        item['videos']['official']=stair_case(policy,folder/'stairs',official=True,video=True)
        persist()
    errors=sum('error' in c or 'full_error' in c for c in state['candidates'].values())
    errors+=sum('error' in case for c in state['candidates'].values() for case in c.get('cases',{}).values())
    errors+=sum('error' in case for c in state['candidates'].values() for case in c.get('videos',{}).values())
    state.update(stage='complete' if not errors else 'complete_with_errors',error_count=errors,finished=time.time());persist()
    atomic_json(HERE/'selection.json',{'recommended_iteration':state['recommended_iteration'],'shortlist':state['shortlist']})
    return state


if __name__=='__main__':
    import sys
    main(sys.argv[1])
