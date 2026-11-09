"""Capacity-gated MoE long run. Usage: python workflow.py [--resume-checkpoint P --diagnosis NOTE]."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import fcntl
from common import HERE,ROOT,GIB,TARGET_ITERATIONS,atomic_json,sha,finite,run_training,validate_training,retain_checkpoint

def resources():
    total,used=map(float,subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(','))
    limit=int(Path('/sys/fs/cgroup/memory.max').read_text());current=int(Path('/sys/fs/cgroup/memory.current').read_text())
    return {'gpu_free':int((total-used)*1024**2),'memory_free':limit-current,'disk_free':shutil.disk_usage(ROOT).free}

def verify_manifest():
    record=json.loads((HERE/'manifest.json').read_text())
    for name,digest in record['files'].items():assert sha(Path(name))==digest,f'Changed source or protected artifact: {name}'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume-checkpoint',type=Path)
    parser.add_argument('--diagnosis');parser.add_argument('--capacity-retry',action='store_true');parser.add_argument('--evaluate-only',action='store_true');args=parser.parse_args()
    lock=(HERE/'workflow.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=HERE/'status.json';state=json.loads(status.read_text()) if status.exists() else {'training_sessions':[],'capacity_runs':[]}
    if args.capacity_retry or args.resume_checkpoint:state.pop('error',None)
    def update(stage,**extra):
        state.update(stage=stage,updated=datetime.now().astimezone().isoformat(),**extra);atomic_json(status,state)
    def wait_for(gpu,memory,disk,reason):
        while True:
            r=resources()
            if r['gpu_free']>=gpu and r['memory_free']>=memory and r['disk_free']>=disk:return r
            update('waiting_for_resources',waiting_reason=reason,required={'gpu':gpu,'memory':memory,'disk':disk},resources=r)
            time.sleep(30)
    try:
        verify_manifest()
        assert json.loads((HERE/'preflight_status.json').read_text())['passed']
        assert json.loads((HERE/'evaluation_smoke.json').read_text())['passed']
        capacity_path=HERE/'capacity_status.json'
        if not capacity_path.exists() or not json.loads(capacity_path.read_text()).get('passed'):
            assert not state['training_sessions'],'Cannot repeat capacity after formal training'
            previous=None;prior=None;selected=4
            stages=[(1024,10),(4096,10),(8192,100)]
            if args.capacity_retry:
                assert args.diagnosis and not state['training_sessions']
                state.setdefault('capacity_diagnoses',[]).append(args.diagnosis)
                prior_runs=[Path(s) for s in state['capacity_runs'] if (Path(s)/'validation.json').exists()]
                assert any(json.loads((p/'audit.json').read_text())['num_envs']==1024 for p in prior_runs)
                assert any(json.loads((p/'audit.json').read_text())['num_envs']==4096 for p in prior_runs)
                selected=8;stages=[(8192,100)]
            for n,iterations in stages:
                required_gpu=4*GIB;required_memory=8*GIB
                if previous:
                    factor=n/previous['n']
                    for key,reserve in [('gpu_increment',2*GIB),('memory_increment',5*GIB)]:
                        slope=max(0,(previous[key]-prior[key])/(previous['n']-prior['n'])) if prior else previous[key]/previous['n']
                        predicted=int((previous[key]+slope*(n-previous['n']))*1.15)+reserve
                        if key=='gpu_increment':required_gpu=max(required_gpu,predicted)
                        else:required_memory=max(required_memory,predicted)
                attempt=0
                while True:
                    before=wait_for(required_gpu,required_memory,20*GIB,f'Capacity {n}, minibatches {selected}')
                    verify_manifest();attempt+=1
                    folder=HERE/f'capacity_{n}_mb{selected}_{time.time_ns()}'
                    update('capacity',current_capacity=str(folder),num_envs=n,mini_batches=selected)
                    code,r=run_training(folder,HERE.name+f'_capacity_{n}',iterations,selected,iterations,num_envs=n)
                    state['capacity_runs'].append(str(folder))
                    log=(folder/'train.log').read_text(errors='replace')
                    audit=json.loads((folder/'audit.json').read_text()) if (folder/'audit.json').exists() else {}
                    memory_problem=('out of memory' in log.lower() or 'headroom exhausted' in log.lower() or (folder/'resource_stop.json').exists())
                    if code!=0:
                        if memory_problem and audit.get('failure_phase',audit.get('phase'))=='optimization' and selected<32:
                            selected*=2;continue
                        if memory_problem:
                            # Retry only after material resource improvement, not in a failure loop.
                            required_gpu=max(required_gpu,before['gpu_free']+2*GIB)
                            required_memory=max(required_memory,before['memory_free'])
                            update('waiting_for_resources',failed_capacity=str(folder),reason='Capacity resource failure; existing jobs preserved')
                            continue
                        raise RuntimeError(f'Capacity failed; diagnose {folder}/train.log')
                    checked=validate_training(folder,iterations,num_envs=n)
                    assert checked['audit']['recovery_guard']['session_incidents']==0,'Capacity had a physics anomaly'
                    assert r['events_after']['oom']==r['events_before']['oom'] and r['events_after']['oom_kill']==r['events_before']['oom_kill']
                    if r['gpu_min_free_mib']<2048 or r['memory_min_free_bytes']<5*GIB:
                        phase=r['phase_peaks']
                        opt=phase.get('optimization',{}).get('gpu_used_mib',0)
                        other=max((v.get('gpu_used_mib',0) for k,v in phase.items() if k!='optimization'),default=0)
                        if r['gpu_min_free_mib']<2048 and opt>r['gpu_total_mib']-2048 and other<=r['gpu_total_mib']-2048 and selected<32:
                            selected*=2;continue
                        required_gpu=max(required_gpu,before['gpu_free']+2*GIB)
                        required_memory=max(required_memory,before['memory_free']+2*GIB)
                        update('waiting_for_resources',failed_capacity=str(folder),reason='Peak headroom insufficient');continue
                    prior=previous
                    previous={'n':n,'gpu_increment':max(0,(r['gpu_peak_used_mib']*1024**2)-(r['gpu_total_mib']*1024**2-before['gpu_free'])),
                              'memory_increment':max(0,r['cgroup_peak_bytes']-(r['memory_limit_bytes']-before['memory_free']))}
                    break
            capacity={'passed':True,'mini_batches':selected,'resources':r,'validation':checked,
                      'formal_required_gpu':int(previous['gpu_increment']*1.1)+2*GIB,
                      'formal_required_memory':int(previous['memory_increment']*1.1)+5*GIB,
                      'estimated_training_hours':checked['mean_steady_iteration_s']*TARGET_ITERATIONS/3600}
            capacity['estimated_disk_bytes']=int(Path(checked['checkpoint']).stat().st_size*361*2.0+5*GIB)
            atomic_json(capacity_path,capacity)
        capacity=json.loads(capacity_path.read_text());batches=capacity['mini_batches']
        if not args.evaluate_only:
            completed=0
            if args.resume_checkpoint:
                assert args.diagnosis,'Diagnose before resuming'
                import torch
                saved=torch.load(args.resume_checkpoint,map_location='cpu',weights_only=False);finite(saved)
                completed=int(saved['iter'])+1;assert 0<completed<TARGET_ITERATIONS
                allowed=[json.loads((Path(s)/'audit.json').read_text()).get('log_dir') for s in state['training_sessions']]
                assert str(args.resume_checkpoint.parent) in allowed,'Only this formal run can resume'
                state.setdefault('diagnoses',[]).append({'checkpoint':str(args.resume_checkpoint),'note':args.diagnosis})
            else:assert not state['training_sessions'],'Use a diagnosed resume for an existing experiment'
            wait_for(capacity['formal_required_gpu'],capacity['formal_required_memory'],capacity['estimated_disk_bytes']+10*GIB,'Formal launch')
            verify_manifest();folder=HERE/('formal' if completed==0 else f'formal_resume_{completed}_{time.time_ns()}')
            state['training_sessions'].append(str(folder));update('training',active_training=str(folder),mini_batches=batches,target_iterations=TARGET_ITERATIONS)
            code,r=run_training(folder,HERE.name+'_formal',TARGET_ITERATIONS-completed,batches,500,
                                fresh=completed==0,checkpoint=args.resume_checkpoint)
            assert code==0,f'Formal training stopped; inspect {folder}/train.log'
            state['training_validation']=validate_training(folder,TARGET_ITERATIONS-completed,TARGET_ITERATIONS-1)
            archive=HERE/'checkpoints';archive.mkdir(exist_ok=True)
            sessions=[json.loads((Path(s)/'audit.json').read_text()) for s in state['training_sessions']]
            import torch
            for index,audit in enumerate(sessions):
                end=sessions[index+1]['start_iteration']-1 if index+1<len(sessions) else TARGET_ITERATIONS-1
                for path in Path(audit['log_dir']).glob('model_*.pt'):
                    if int(path.stem.split('_')[1])<=end:
                        finite(torch.load(path,map_location='cpu',weights_only=False));retain_checkpoint(path,archive/path.name)
            shutil.copytree(Path(sessions[0]['log_dir'])/'params',archive/'params',dirs_exist_ok=True)
            for i in [*range(500,TARGET_ITERATIONS,500),TARGET_ITERATIONS-1]:assert (archive/f'model_{i}.pt').exists()
            state['checkpoint']=str(archive/f'model_{TARGET_ITERATIONS-1}.pt');update('training_complete')
        verify_manifest();update('evaluating')
        from evaluate import main as evaluate
        result=evaluate(state['checkpoint']);state['recommended_iteration']=result['recommended_iteration']
        from report import create_report
        create_report(state,result)
        update('complete' if result['error_count']==0 else 'complete_with_delivery_errors',joint_acceptance=result['recommended_iteration'] is not None)
    except BaseException:
        update('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
