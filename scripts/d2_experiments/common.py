"""Local experiment support: resource monitoring and real training validation."""
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import threading
import time

from _paths import HERE, ROOT
GIB = 1024**3
CONFIG = json.loads((HERE/'experiment.json').read_text())
TARGET_ITERATIONS = int(CONFIG['target_iterations'])


def retain_checkpoint(source, destination):
    """Retain immutable experiment checkpoints without duplicating their disk blocks."""
    source,destination=Path(source),Path(destination)
    if destination.exists():
        assert sha(source)==sha(destination),'Conflicting checkpoint archive'
        return
    try:os.link(source,destination)
    except OSError:shutil.copy2(source,destination)


def atomic_json(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    import torch
    if isinstance(value,torch.Tensor):
        if not torch.isfinite(value).all():raise FloatingPointError('Non-finite tensor')
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite(v)
    elif isinstance(value,float) and not math.isfinite(value):
        raise FloatingPointError('Non-finite scalar')


def exact_state(left, right):
    """Compare complete model/Adam state, including moments and parameter groups."""
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact_state(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return isinstance(right, (list, tuple)) and len(left) == len(right) and all(exact_state(a, b) for a, b in zip(left, right))
    return left == right


def cgroup_events():
    return {k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.events').read_text().splitlines())}


class Monitor:
    def __init__(self, folder):
        self.folder=Path(folder);self.lock=threading.Lock();self.stop_event=threading.Event()
        self.data={'gpu_peak_used_mib':0,'gpu_min_free_mib':1e12,'gpu_samples':0,
                   'memory_limit_bytes':int(Path('/sys/fs/cgroup/memory.max').read_text()),
                   'cgroup_peak_bytes':0,'anonymous_peak_bytes':0,'memory_samples':0,
                   'events_before':cgroup_events(),'phase_peaks':{}}
        self.gpu=subprocess.Popen(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits','--loop-ms=200'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.threads=[threading.Thread(target=self._gpu,daemon=True),threading.Thread(target=self._memory,daemon=True)]
        for thread in self.threads:thread.start()

    def _phase(self):
        try:return json.loads((self.folder/'audit.json').read_text()).get('phase','unknown')
        except (FileNotFoundError,json.JSONDecodeError):return 'launch'

    def _gpu(self):
        for line in self.gpu.stdout:
            try:total,used,util=map(float,line.strip().split(','))
            except ValueError:continue
            phase=self._phase()
            with self.lock:
                self.data['gpu_total_mib']=total
                self.data['gpu_peak_used_mib']=max(self.data['gpu_peak_used_mib'],used)
                self.data['gpu_min_free_mib']=min(self.data['gpu_min_free_mib'],total-used)
                self.data['gpu_samples']+=1
                entry=self.data['phase_peaks'].setdefault(phase,{})
                entry['gpu_used_mib']=max(entry.get('gpu_used_mib',0),used)

    def _memory(self):
        with (self.folder/'resources.jsonl').open('a') as file:
            tick=0
            while not self.stop_event.is_set():
                current=int(Path('/sys/fs/cgroup/memory.current').read_text())
                stat=dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
                phase=self._phase()
                with self.lock:
                    self.data['cgroup_peak_bytes']=max(self.data['cgroup_peak_bytes'],current)
                    self.data['anonymous_peak_bytes']=max(self.data['anonymous_peak_bytes'],int(stat['anon']))
                    self.data['memory_samples']+=1
                    self.data['phase_peaks'].setdefault(phase,{})['cgroup_bytes']=max(
                        self.data['phase_peaks'].get(phase,{}).get('cgroup_bytes',0),current)
                    if tick%4==0:
                        snapshot={**self.data,'time':time.time(),'phase':phase,'current_cgroup_bytes':current}
                        file.write(json.dumps(snapshot)+'\n');file.flush()
                        atomic_json(self.folder/'resources.json',snapshot)
                tick+=1;self.stop_event.wait(.25)

    def finish(self):
        self.stop_event.set();self.gpu.terminate()
        try:self.gpu.wait(timeout=5)
        except subprocess.TimeoutExpired:self.gpu.kill();self.gpu.wait()
        for thread in self.threads:thread.join(timeout=5)
        self.data['events_after']=cgroup_events()
        self.data['memory_min_free_bytes']=self.data['memory_limit_bytes']-self.data['cgroup_peak_bytes']
        atomic_json(self.folder/'resources.json',self.data)
        return self.data


def run_training(folder, experiment, iterations, batches, save_interval, *, fresh=True, checkpoint=None, num_envs=8192, recovery=True, injection=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for filename in ['audit_train.py','common.py','recovery_guard.py']:
        shutil.copy2(HERE/filename,folder/(Path(filename).stem+'_used.py'))
    command=[os.sys.executable,str(HERE/('guard_injection_train.py' if injection else 'audit_train.py')),'--task','d2-mixed-moe-cts','--physics','newton_mjwarp',
        '--viz','none','--num_envs',str(num_envs),'--max_iterations',str(iterations),'--seed','42',
        '--experiment_name',experiment,'--run_name','zero10_scratch','--logger','tensorboard','--log_interval','1',
        'env.commands.base_velocity.limit_ang_vel_at_zero_command_prob=0.0',f'agent.save_interval={save_interval}',f'agent.algorithm.num_mini_batches={batches}']
    if not fresh:command+=['--resume','--checkpoint',str(checkpoint)]
    atomic_json(folder/'command.json',command)
    env=os.environ.copy();env['D2_AUDIT_DIR']=str(folder);env['D2_REQUIRE_FRESH']='1' if fresh else '0'
    env['D2_ALLOW_RECOVERY']='1' if recovery else '0'
    if injection:env['D2_GUARD_TEST_MODE']=injection
    env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    monitor=Monitor(folder)
    started=time.monotonic()
    try:
        with (folder/'train.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(folder/'process.json',{'pid':process.pid,'started':datetime.now().astimezone().isoformat()})
            while process.poll() is None:
                current=int(Path('/sys/fs/cgroup/memory.current').read_text())
                limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
                if limit-current<5*GIB or shutil.disk_usage(ROOT).free<10*GIB or monitor.data['gpu_min_free_mib']<2048:
                    atomic_json(folder/'resource_stop.json',{'memory_free':limit-current,'disk_free':shutil.disk_usage(ROOT).free,'gpu_min_free_mib':monitor.data['gpu_min_free_mib']})
                    process.terminate()
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    break
                time.sleep(.5)
            code=process.wait()
    finally:
        resources=monitor.finish()
    atomic_json(folder/'process_result.json',{'exit_code':code,'wall_s':time.monotonic()-started})
    return code,resources


def validate_training(folder, expected_iterations, expected_final_iter=None, num_envs=8192):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    torch.set_num_threads(1)
    folder=Path(folder);audit=json.loads((folder/'audit.json').read_text())
    assert audit['phase']=='complete',audit
    assert audit['policy_steps']==expected_iterations*24
    assert audit['total_env_steps']==num_envs*24*expected_iterations
    assert audit['num_envs']==num_envs
    checkpoint=Path(audit['last_checkpoint'])
    initial=torch.load(folder/'initial_state.pt',map_location='cpu',weights_only=False)
    trained=torch.load(checkpoint,map_location='cpu',weights_only=False)
    finite(trained)
    assert trained['optimizer_stu_enc_state_dict']['state'], 'Student Adam state missing'
    for key in ['optimizer_state_dict','optimizer_stu_enc_state_dict']:
        before=initial[key]['state'];after=trained[key]['state']
        for ident,item in after.items():
            assert float(item['step']) > float(before.get(ident,{}).get('step',0))
    assert trained['iter']==(expected_final_iter if expected_final_iter is not None else expected_iterations-1)
    assert trained['policy_identity']==initial['policy_identity']
    updates={}
    for prefix in ['actor.','critic.','teacher_encoder.','student_moe_encoder.moe.experts.','student_moe_encoder.moe.gating_network.']:
        keys=[k for k in initial['model_state_dict'] if k.startswith(prefix) and k.endswith('weight')]
        updates[prefix]=sum((trained['model_state_dict'][k]-initial['model_state_dict'][k]).square().sum().item() for k in keys)**.5
        assert updates[prefix]>0
    expert_key='student_moe_encoder.moe.experts.experts.weight'
    delta=(trained['model_state_dict'][expert_key]-initial['model_state_dict'][expert_key]).reshape(8,-1)
    updates['individual_experts']=delta.norm(dim=1).tolist()
    assert all(v>0 for v in updates['individual_experts']), 'An expert head did not update'
    events=EventAccumulator(str(checkpoint.parent),size_guidance={'scalars':0}).Reload()
    tags=events.Tags()['scalars'];assert any(k.startswith('Loss/') for k in tags)
    last={}
    for tag in tags:
        values=events.Scalars(tag);assert all(math.isfinite(v.value) for v in values),tag
        if 'non_finite' in tag.lower():
            if audit.get('recovery_enabled'):
                assert all(0<=v.value<=1/num_envs+1e-9 for v in values),tag
            else:assert max(v.value for v in values)==0,tag
        last[tag]=values[-1].value
    rows=[json.loads(line) for line in (folder/'iterations.jsonl').read_text().splitlines()]
    stable=[r['iteration_wall_s'] for r in (rows[-50:]) if r['iteration_wall_s'] is not None]
    result={'finite':True,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'audit':audit,
            'updates_l2':updates,'final_scalars':last,'mean_steady_iteration_s':sum(stable)/len(stable) if stable else None,
            'initial_state_sha256':sha(folder/'initial_state.pt')}
    if audit.get('recovery_enabled'):
        recovery=audit['recovery_guard']
        assert recovery['pending'] is None
        assert recovery['session_incidents']==recovery['session_recovered']
        assert recovery['total_incidents']==recovery['total_recovered']
        assert recovery['optimizer_steps_checked']==expected_iterations*audit['mini_batches']*5
        assert recovery['student_optimizer_steps_checked']==expected_iterations*audit['mini_batches']*5
        incidents=[json.loads(line) for line in (folder/'recovery_events.jsonl').read_text().splitlines()]
        assert sum(x['kind']=='recovery_requested' for x in incidents)==recovery['session_incidents']
        assert sum(x['kind']=='recovery_confirmed' for x in incidents)==recovery['session_recovered']
        assert trained['infos']['recovery_guard']['pending'] is None
        result['recovery_validation']=recovery
    atomic_json(folder/'validation.json',result)
    return result
