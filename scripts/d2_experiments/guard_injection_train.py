"""Synthetic tests only: D2_GUARD_TEST_MODE=single/repeat/multiple/gradient/recovery_failure python guard_injection_train.py <train args>."""
from pathlib import Path
import os,runpy,sys
from _paths import HERE, ROOT
assert 'guard_test_' in os.environ['D2_AUDIT_DIR'],'Injection is restricted to isolated test folders'
MODE=os.environ['D2_GUARD_TEST_MODE']
assert MODE in {'single','repeat','multiple','gradient','student_gradient','recovery_failure'}
sys.path.insert(0,str(ROOT/'scripts/rsl_rl'))
import runtime_args
original_bootstrap=runtime_args.bootstrap_runtime

def bootstrap(*args,**kwargs):
    original_bootstrap(*args,**kwargs)
    import torch,warp as wp
    from rsl_rl.base.base_runner import BaseRunner
    from rsl_rl.base.env import RslRlVecEnvWrapper
    from isaaclab_newton.physics.newton_manager import NewtonManager
    original_find=RslRlVecEnvWrapper._find_nonfinite_envs
    original_init=BaseRunner.__init__;original_learn=BaseRunner.learn
    calls=0
    def find(env,obs,rewards):
        nonlocal calls
        calls+=1
        if MODE not in {'gradient','student_gradient'} and (calls==3 or (MODE=='repeat' and calls==4)):
            ids=[3,4] if MODE=='multiple' else [3]
            for _,value in obs.items(include_nested=True,leaves_only=True):value[ids]=float('nan')
            rewards[ids]=float('nan')
            for name in ['qpos','qvel','qacc','qacc_warmstart']:
                wp.to_torch(getattr(NewtonManager._solver.mjw_data,name))[ids]=float('nan')
        return original_find(env,obs,rewards)
    def initialize(runner,*args,**kwargs):
        original_init(runner,*args,**kwargs)
        if MODE=='recovery_failure':
            original_physics_step=runner.env.unwrapped.sim.step
            def physics_step(*a,**kw):
                original_physics_step(*a,**kw)
                if calls>=3:wp.to_torch(NewtonManager._solver.mjw_data.qacc)[3]=float('nan')
            runner.env.unwrapped.sim.step=physics_step
    def learn(runner,*args,**kwargs):
        if MODE=='gradient':next(runner.alg.policy.teacher_encoder.parameters()).register_hook(lambda grad:grad*float('nan'))
        if MODE=='student_gradient':next(runner.alg.policy.student_moe_encoder.parameters()).register_hook(lambda grad:grad*float('nan'))
        return original_learn(runner,*args,**kwargs)
    RslRlVecEnvWrapper._find_nonfinite_envs=find;BaseRunner.__init__=initialize;BaseRunner.learn=learn
runtime_args.bootstrap_runtime=bootstrap
runpy.run_path(str(HERE/'audit_train.py'),run_name='__main__')
