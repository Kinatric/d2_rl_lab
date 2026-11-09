"""Production trainer with passive audit and finite checks; set D2_AUDIT_DIR and pass train CLI args."""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

from _paths import ROOT
OUT = Path(os.environ['D2_AUDIT_DIR'])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT/'scripts/rsl_rl'))
state = {'pid': os.getpid(), 'policy_steps': 0, 'iterations_completed': 0}
counts = None
previous_log_time = None
guard = None
runner_ref = None
ALLOW_RECOVERY = os.environ.get('D2_ALLOW_RECOVERY') == '1'


def publish(phase, **extra):
    state.update(phase=phase, updated=datetime.now().astimezone().isoformat(), **extra)
    tmp = OUT/'audit.tmp'
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(OUT/'audit.json')


publish('runtime_bootstrap')
import runtime_args
original_bootstrap = runtime_args.bootstrap_runtime


def bootstrap(*args, **kwargs):
    original_bootstrap(*args, **kwargs)
    import torch
    from rsl_rl.base.base_runner import BaseRunner
    from rsl_rl.base.env import RslRlVecEnvWrapper
    from rsl_rl.moe_cts.algorithm import MoECTS as PPO
    from rsl_rl.utils.logger import Logger
    from common import finite
    from recovery_guard import Guard, fast_finite

    original_init, original_learn, original_save = BaseRunner.__init__, BaseRunner.learn, BaseRunner.save
    original_update, original_step, original_log = PPO.update, RslRlVecEnvWrapper.step, Logger.log
    original_find = RslRlVecEnvWrapper._find_nonfinite_envs
    original_load = BaseRunner.load

    def load(runner, *positional, **keywords):
        infos = original_load(runner, *positional, **keywords)
        # Validate both optimizers before allowing any new rollout or update.
        from common import exact_state
        checkpoint_path = positional[0] if positional else keywords['path']
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        loaded = runner._checkpoint_state(None)
        keys = ['model_state_dict', 'optimizer_state_dict', 'optimizer_stu_enc_state_dict', 'policy_identity']
        checks = {key: exact_state(saved[key], loaded[key]) for key in keys}
        assert all(checks.values()), f'Incomplete checkpoint restore: {checks}'
        state['exact_restore_checks'] = checks
        runner._recovery_restored = infos.get('recovery_guard') if isinstance(infos, dict) else None
        return infos

    def diagnose_nonfinite(env, observations, rewards):
        mask,sources=original_find(env,observations,rewards)
        if guard is not None:
            physical_mask,physical_sources=guard.bad_mask()
            mask |= physical_mask
            sources.update(physical_sources)
            if mask.any().item():
                guard.anomaly(mask.nonzero(as_tuple=False).flatten().tolist(),observations,rewards,sources)
            return mask,sources
        if mask.any().item():
            ids=mask.nonzero(as_tuple=False).flatten()
            incident={'environment_ids':ids.cpu(),'common_step_counter':int(env.unwrapped.common_step_counter),
                'sources':{k:v[ids].cpu() for k,v in sources.items()},
                'observations':{str(k):v[ids].detach().cpu() for k,v in observations.items(include_nested=True,leaves_only=True)},
                'rewards':rewards[ids].detach().cpu(),'robot':{}}
            for name in ['joint_pos','joint_vel','root_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w']:
                try:
                    value=getattr(env.unwrapped.scene['robot'].data,name)
                    value=value.torch if hasattr(value,'torch') else value
                    incident['robot'][name]=value[ids].detach().cpu()
                except Exception as error:incident['robot'][name]=str(error)
            torch.save(incident,OUT/f'nonfinite_step_{incident["common_step_counter"]}.pt')
        return mask,sources

    def initialize(runner, *positional, **keywords):
        publish('runner_initialization')
        original_init(runner, *positional, **keywords)
        initial = runner._checkpoint_state(None)
        finite(initial)
        torch.save(initial, OUT/'initial_state.pt')
        state.update(log_dir=str(runner.logger.log_dir), num_envs=runner.env.num_envs,
                     steps_per_iteration=runner.cfg['num_steps_per_env'],
                     mini_batches=runner.alg.num_mini_batches, initial_iteration=runner.current_learning_iteration,
                     initialization_complete=datetime.now().astimezone().isoformat())
        publish('runner_ready')

    def learn(runner, *positional, **keywords):
        global guard,runner_ref
        runner_ref=runner
        if os.environ.get('D2_REQUIRE_FRESH', '1') == '1':
            assert runner.current_learning_iteration == 0
            assert runner.env.unwrapped.common_step_counter == 0
            assert not runner.alg.optimizer.state, 'Fresh training unexpectedly has optimizer history'
            assert not runner.alg.optimizer_stu_enc.state, 'Student optimizer was not fresh'
        else:
            assert runner.current_learning_iteration>0 and runner.alg.optimizer.state
            assert runner.env.unwrapped.common_step_counter==runner.current_learning_iteration*24
            rates={float(group['lr']) for group in runner.alg.optimizer.param_groups}
            assert len(rates)==1,'Unexpected mixed optimizer learning rates'
            runner.alg.learning_rate=rates.pop()
            state['restored_student_learning_rates']=[float(g['lr']) for g in runner.alg.optimizer_stu_enc.param_groups]
            state['restored_learning_rate']=runner.alg.learning_rate
            finite(runner._checkpoint_state(None))
            torch.save(runner._checkpoint_state(None),OUT/'initial_state.pt')
        state['start_iteration'] = runner.current_learning_iteration
        state['start_common_step_counter'] = int(runner.env.unwrapped.common_step_counter)
        if ALLOW_RECOVERY:
            guard=Guard(runner,OUT,getattr(runner,'_recovery_restored',None))
            state['recovery_enabled']=True
        result=original_learn(runner, *positional, **keywords)
        if guard is not None:
            assert guard.ledger.pending is None,'Final recovery was not verified before training ended'
            state['recovery_guard']=guard.summary()
        return result

    def step(env, actions):
        global counts
        if state['policy_steps'] % 24 == 0:
            publish('sampling')
        if guard is not None:
            fast_finite(actions)
            guard.current_step=int(env.unwrapped.common_step_counter)+1
        command = env.unwrapped.command_manager.get_command('base_velocity')
        if counts is None:
            counts = torch.zeros(3, dtype=torch.int64, device=command.device)
        counts[0] += (command == 0).all(dim=1).sum()
        counts[1] += command.shape[0]
        counts[2] += ((command[:, :2] == 0).all(dim=1) & (command[:, 2] != 0)).sum()
        result = original_step(env, actions)
        state['policy_steps'] += 1
        state['group_metrics']=group_metrics(env, result, command)
        bad = result[3].get('log', {}).get('Episode_Termination/non_finite', 0.)
        if guard is not None:
            guard.finish_step(result)
            state['recovery_guard']=guard.summary()
        elif float(bad) != 0.:
            raise FloatingPointError('Non-finite environment state detected')
        state['common_step_counter'] = int(env.unwrapped.common_step_counter)
        state['zero_command_curriculum_probability'] = float(env.unwrapped.command_manager.get_term('base_velocity').zero_command_prob)
        return result

    def group_metrics(env, result, command):
        groups={'teacher':runner_ref.alg.teacher_env_idxs,'student':runner_ref.alg.student_env_idxs}
        output={}
        raw=env.unwrapped
        velocity=raw.scene['robot'].data.root_lin_vel_b
        velocity=velocity.torch if hasattr(velocity,'torch') else velocity
        for name,ids in groups.items():
            item={'reward':float(result[1][ids].mean()),'done_fraction':float(result[2][ids].float().mean()),
                  'zero_fraction':float((command[ids]==0).all(dim=1).float().mean()),
                  'velocity_xy_rmse':float((velocity[ids,:2]-command[ids,:2]).square().mean().sqrt())}
            levels=getattr(raw.scene.terrain,'terrain_levels',None)
            if levels is not None:item['terrain_level']=float(levels[ids].float().mean())
            item['terminations']={n:float(raw.termination_manager.get_term(n)[ids].float().mean()) for n in raw.termination_manager.active_terms}
            output[name]=item
        return output

    def update(algorithm):
        publish('optimization')
        losses = original_update(algorithm)
        with torch.no_grad():
            obs=runner_ref.env.get_observations()
            actor=algorithm.policy.actor_obs_normalizer(algorithm.policy.get_actor_obs(obs))
            _,weights=algorithm.policy.student_moe_encoder(actor[algorithm.student_env_idxs[:256]])
            state['expert_mean_usage']=weights.mean(dim=0).cpu().tolist()
        finite(losses)
        return losses

    def log(logger, *positional, **keywords):
        global previous_log_time
        finite(keywords['loss_dict'])
        original_log(logger, *positional, **keywords)
        now = time.monotonic()
        zero, total, yaw = counts.tolist()
        record = {'iterations_completed': int(keywords['it']), 'zero_command_env_steps': zero,
                  'total_env_steps': total, 'zero_command_time_fraction': zero/total,
                  'yaw_only_env_steps': yaw, 'losses': keywords['loss_dict'],
                  'learning_rate': keywords['learning_rate'],
                  'zero_command_curriculum_probability': state['zero_command_curriculum_probability'],
                  'pytorch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'pytorch_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'collection_s': keywords['collect_time'], 'learning_s': keywords['learn_time'],
                  'iteration_wall_s': now-previous_log_time if previous_log_time is not None else None}
        if guard is not None:
            record['group_metrics']=state.get('group_metrics',{})
            record['expert_mean_usage']=state.get('expert_mean_usage',[])
            record['student_learning_rates']=[float(g['lr']) for g in runner_ref.alg.optimizer_stu_enc.param_groups]
            record['recovery_guard']=guard.summary()
            record['resources']=guard.check_resources()
        previous_log_time = now
        publish('iteration_complete', **record)
        with (OUT/'iterations.jsonl').open('a') as file:
            file.write(json.dumps({**record, 'common_step_counter':state['common_step_counter']})+'\n')

    def save(runner, path, *positional, **keywords):
        publish('checkpoint_save')
        finite(runner._checkpoint_state(None))
        if guard is not None:
            assert not positional,'Unexpected positional save infos'
            infos=dict(keywords.pop('infos',None) or {})
            infos['recovery_guard']=guard.summary()
            keywords['infos']=infos
        temporary=str(path)+'.partial'
        original_save(runner, temporary, *positional, **keywords)
        Path(temporary).replace(path)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        finite(saved)
        assert saved['iter']==runner.current_learning_iteration
        assert 'optimizer_stu_enc_state_dict' in saved
        del saved
        publish('checkpoint_saved', last_checkpoint=str(path))

    BaseRunner.__init__, BaseRunner.learn, BaseRunner.save = initialize, learn, save
    PPO.update, RslRlVecEnvWrapper.step, Logger.log = update, step, log
    RslRlVecEnvWrapper._find_nonfinite_envs = diagnose_nonfinite
    BaseRunner.load = load
    publish('scene_initialization')


runtime_args.bootstrap_runtime = bootstrap
sys.argv[0] = str(ROOT/'scripts/rsl_rl/train.py')
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
    publish('complete')
except BaseException:
    if guard is not None:
        state['recovery_guard']=guard.summary()
        try:
            import torch
            torch.save(runner_ref._checkpoint_state(None),OUT/'failed_learner_state.pt')
        except Exception as error:state['failure_state_capture_error']=str(error)
    failure_phase = state.get('phase')
    publish('failed', failure_phase=failure_phase, error=traceback.format_exc())
    raise
