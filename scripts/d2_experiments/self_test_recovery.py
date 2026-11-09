"""Check escalation windows and learner finite guards: python self_test_recovery.py."""
import json
import torch
from recovery_guard import Ledger, RecoveryStop, fast_finite

def rejects(function):
    try: function()
    except (RecoveryStop, RuntimeError): return
    raise AssertionError('Expected a hard stop')

one=Ledger();one.begin(10,[7]);one.pending['physics_confirmed']=True;one.recovered()
assert one.total_recovered==1 and one.pending is None
restored=Ledger(json.loads(json.dumps(one.state())))
rejects(lambda:restored.begin(20,[7]))
late=Ledger(one.state());late.begin(2411,[7]);late.pending['physics_confirmed']=True;late.recovered()
multi=Ledger();rejects(lambda:multi.begin(1,[7,1]))
spread=Ledger()
for step,i in [(10,1),(11,2)]:spread.begin(step,[i]);spread.pending['physics_confirmed']=True;spread.recovered()
rejects(lambda:spread.begin(12,[3]))
pending=Ledger();pending.begin(10,[1]);rejects(lambda:pending.begin(11,[2]))
fast_finite({'weights':torch.tensor([1.,-3.]),'optimizer':{'step':torch.tensor(2.)}})
for value in [float('nan'),float('inf'),float('-inf')]:
    rejects(lambda:fast_finite({'model':torch.tensor([value])}))
    rejects(lambda:fast_finite({'loss':value}))
fast_finite(torch.tensor([3e38,3e38]))
print(json.dumps({'passed':True,'cases':['isolated_recovery','cross_resume_repeat','expired_window','simultaneous_spread','rolling_frequency','pending_recovery','tensor_and_scalar_nonfinite','finite_norm_overflow_avoided']}))
