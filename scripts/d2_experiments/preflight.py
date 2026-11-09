"""Run bounded MoE integration and injected-fault checks: python preflight.py."""
import json
import traceback
from common import HERE, run_training, validate_training, atomic_json

state={'stage':'smoke','passed':False,'cases':{}}
def publish(): atomic_json(HERE/'preflight_status.json',state)

try:
    publish()
    folder=HERE/'smoke32'
    code,resources=run_training(folder,HERE.name+'_smoke',2,4,1,num_envs=32)
    assert code==0, f'Smoke failed: {folder}/train.log'
    first=validate_training(folder,2,num_envs=32)
    state['cases']['smoke']=first;publish()
    folder=HERE/'resume32'
    code,resources=run_training(folder,HERE.name+'_resume',1,4,1,num_envs=32,
        fresh=False,checkpoint=first['checkpoint'])
    assert code==0,f'Resume failed: {folder}/train.log'
    resumed=validate_training(folder,1,2,num_envs=32)
    state['cases']['resume']=resumed;publish()
    from evaluate import export_checkpoint
    policy=export_checkpoint(__import__('pathlib').Path(resumed['checkpoint']),HERE/'smoke_export')
    state['student_export']=str(policy);publish()
    for mode,expected in [('single',None),('repeat','Another anomaly'),('multiple','Multiple environments'),
                          ('gradient','non-finite'),('student_gradient','non-finite'),
                          ('recovery_failure','Physical state was not finite')]:
        state.update(stage='fault_test',current=mode);publish()
        folder=HERE/f'guard_test_{mode}'
        code,_=run_training(folder,HERE.name+'_fault_'+mode,2,4,1,num_envs=32,injection=mode)
        audit=json.loads((folder/'audit.json').read_text())
        if expected is None:
            assert code==0,audit
            checked=validate_training(folder,2,num_envs=32)
            assert checked['recovery_validation']['session_recovered']==1
        else:
            assert code!=0 and audit['phase']=='failed',audit
            assert expected.lower() in audit['error'].lower(),audit['error']
        state['cases'][mode]={'passed':True,'exit_code':code};publish()
    state.update(stage='passed',passed=True);publish()
except BaseException:
    state.update(stage='failed',error=traceback.format_exc());publish();raise
