"""Exercise export, scoring, stairs, and video before formal launch: python evaluation_smoke.py."""
import json
import traceback
from pathlib import Path
from common import HERE,atomic_json,validate_training
from evaluate import check_exports,evaluate_case,stair_case

try:
    preflight=json.loads((HERE/'preflight_status.json').read_text());assert preflight['passed']
    validate_training(HERE/'smoke32',2,num_envs=32)
    validate_training(HERE/'resume32',1,2,num_envs=32)
    policy=Path(preflight['student_export'])
    checked=check_exports(policy.parent)
    root=HERE/'evaluation_smoke'
    flat=evaluate_case(policy,root,('stand_d2','stand',2,[0,0,0]))
    walk=evaluate_case(policy,root,('forward_d2','forward',2,[.5,0,0]))
    standard=stair_case(policy,root/'stairs',video=True)
    official=stair_case(policy,root/'stairs',official=True)
    for item in [flat,walk,standard,official]:assert 'error' not in item,item
    assert standard['physics_steps']>0 and official['physics_steps']>0
    from score import phase_score
    import numpy as np
    # Stable synthetic standing must pass; a 20 cm drift must fail.
    samples=np.zeros((6000,21));samples[:,0]=np.arange(1,6001)*.02;samples[:,3]=.30;samples[:,9]=1
    metadata={'first_fall_s':None,'warnings':0}
    assert phase_score(samples,0,120,[0,0,0],metadata)['passed']
    samples[:,1]=np.linspace(0,.2,6000)
    assert 'stand_drift' in phase_score(samples,0,120,[0,0,0],metadata)['failures']
    atomic_json(HERE/'evaluation_smoke.json',{'passed':True,'export':checked,'flat':flat,'walk':walk,'stairs':standard,'official':official})
except BaseException:
    atomic_json(HERE/'evaluation_smoke.json',{'passed':False,'error':traceback.format_exc()});raise
