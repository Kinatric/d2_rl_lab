"""Experiment-only isolated recovery and evidence. Used by audit_train.py; test with self_test_recovery.py."""
from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path

LIMITS = {'max_bad_per_step': 1, 'window_policy_steps': 2400,
          'max_incidents_in_window': 2, 'repeat_same_env_within_window': False,
          'trace_physics_steps': 16}


class RecoveryStop(FloatingPointError):
    pass


class Ledger:
    """Absolute policy-step windows survive checkpoint resumes; rejected events are retained."""
    def __init__(self, restored=None):
        restored = restored or {}
        self.recent = deque(restored.get('recent', []))
        self.total_incidents = restored.get('total_incidents', 0)
        self.total_recovered = restored.get('total_recovered', 0)
        self.pending = None
        self.session_incidents = 0
        self.session_recovered = 0

    def begin(self, step, ids):
        self.total_incidents += 1
        self.session_incidents += 1
        if self.pending is not None:
            raise RecoveryStop('Another anomaly before previous recovery was confirmed')
        while self.recent and step - self.recent[0]['step'] > LIMITS['window_policy_steps']:
            self.recent.popleft()
        if len(ids) > LIMITS['max_bad_per_step']:
            raise RecoveryStop('Multiple environments failed on the same policy step')
        if any(set(ids) & set(e['ids']) for e in self.recent):
            raise RecoveryStop('Same environment failed again within 100 iterations')
        if len(self.recent) >= LIMITS['max_incidents_in_window']:
            raise RecoveryStop('Third incident within 100 iterations')
        event = {'step': step, 'ids': list(ids)}
        self.recent.append(event)
        self.pending = {**event, 'physics_confirmed': False}

    def recovered(self):
        assert self.pending and self.pending['physics_confirmed']
        self.pending = None
        self.total_recovered += 1
        self.session_recovered += 1

    def state(self):
        return {'limits': LIMITS, 'recent': list(self.recent), 'pending': self.pending,
                'total_incidents': self.total_incidents, 'total_recovered': self.total_recovered,
                'session_incidents': self.session_incidents, 'session_recovered': self.session_recovered}


def fast_finite(values):
    """Group reductions on each device; reject non-finite tensors without overflowing a norm."""
    import torch
    groups = {}
    def visit(value):
        if isinstance(value, torch.Tensor):
            if value.numel() and (value.is_floating_point() or value.is_complex()):
                groups.setdefault((value.device, value.dtype), []).append(value.detach())
        elif isinstance(value, dict):
            for item in value.values(): visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value: visit(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise RecoveryStop('Non-finite learner scalar')
    visit(values)
    for tensors in groups.values():
        # Infinity norms do not overflow just because the tensor has many finite entries.
        norms = torch._foreach_norm(tensors, float('inf'))
        if not torch.isfinite(torch.stack(norms)).all().item():
            raise RecoveryStop('Non-finite learner tensor')


class Guard:
    CORE = ('qpos', 'qvel', 'qacc', 'qacc_warmstart')
    TRACE = CORE + ('qfrc_applied', 'qfrc_constraint', 'qfrc_bias', 'qM', 'qLD', 'qLDiagInv')

    def __init__(self, runner, out, restored=None):
        import torch
        import warp as wp
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from robot_rl_core.physics.resettable_mjwarp_manager import NewtonMJWarpResettableManager
        self.torch, self.wp = torch, wp
        self.manager, self.reset_manager = NewtonManager, NewtonMJWarpResettableManager
        self.runner, self.env, self.out = runner, runner.env.unwrapped, Path(out)
        self.ledger = Ledger(restored)
        self.history = deque(maxlen=LIMITS['trace_physics_steps'])
        self.physics_steps = 0
        self.optimizer_steps_checked = 0
        self.student_optimizer_steps_checked = 0
        self.current_step = int(self.env.common_step_counter)
        self.data = NewtonManager._solver.mjw_data
        self.views = {k: wp.to_torch(getattr(self.data, k)) for k in self.CORE}
        self.trace_views = {}
        self.layout = {}
        offset = 0
        for name in self.TRACE:
            value = getattr(self.data, name, None)
            if value is None: continue
            tensor = wp.to_torch(value)
            if tensor.shape[0] != runner.env.num_envs: continue
            self.trace_views[name] = tensor
            shape = list(tensor.shape[1:]); size = tensor[0].numel()
            self.layout[name] = {'slice': [offset, offset + size], 'shape': shape}
            offset += size
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out/'recovery_policy.json').write_text(json.dumps(LIMITS, indent=2))
        original_step = self.env.sim.step

        def physics_step(*args, **kwargs):
            original_step(*args, **kwargs)
            self.physics_steps += 1
            self.snapshot()
            pending = self.ledger.pending
            if pending:
                ids = pending['ids']
                good = all(torch.isfinite(v[ids]).all().item() for v in self.views.values())
                if not good or self.reset_manager._has_pending_hard_reset:
                    self.save_incident(ids, 'hard_reset_failed')
                    self.event('hard_reset_failed', ids=ids)
                    raise RecoveryStop('Physical state was not finite after the queued hard reset')
                pending['physics_confirmed'] = True
        self.env.sim.step = physics_step

        # Per-minibatch guard: no invalid gradient norm may enter Adam; check resulting tensors too.
        original_clip = torch.nn.utils.clip_grad_norm_
        def checked_clip(*args, **kwargs):
            kwargs['error_if_nonfinite'] = True
            return original_clip(*args, **kwargs)
        torch.nn.utils.clip_grad_norm_ = checked_clip
        def protect(optimizer, counter):
            original_step=optimizer.step
            def step(*args,**kwargs):
                fast_finite([p.grad for g in optimizer.param_groups for p in g['params'] if p.grad is not None])
                result=original_step(*args,**kwargs)
                fast_finite([runner.alg.policy.state_dict(),optimizer.state])
                setattr(self,counter,getattr(self,counter)+1)
                return result
            optimizer.step=step
        protect(runner.alg.optimizer,'optimizer_steps_checked')
        protect(runner.alg.optimizer_stu_enc,'student_optimizer_steps_checked')
        self.event('guard_started', restored=restored, trace_layout=self.layout)

    def event(self, kind, **extra):
        record = {'time': datetime.now().astimezone().isoformat(), 'kind': kind,
                  'policy_step': self.current_step, 'physics_step': self.physics_steps, **extra}
        with (self.out/'recovery_events.jsonl').open('a') as file:
            file.write(json.dumps(record)+'\n')

    def snapshot(self):
        # Compact rolling state for every world; detailed solver fields are captured on incident.
        self.history.append({'physics_step':self.physics_steps,'policy_step':self.current_step,
            'qpos':self.views['qpos'].detach().clone(),'qvel':self.views['qvel'].detach().clone(),
            'action':self.env.action_manager.action.detach().clone()})

    def bad_mask(self):
        sources = {'physics/'+k: ~self.torch.isfinite(v).reshape(v.shape[0], -1).all(dim=1)
                   for k, v in self.views.items()}
        return self.torch.stack(list(sources.values())).any(dim=0), sources

    def save_incident(self, ids, reason, observations=None, rewards=None, sources=None):
        import random
        import numpy as np
        torch = self.torch
        payload = {'reason': reason, 'environment_ids': list(ids), 'policy_step': self.current_step,
                   'physics_step': self.physics_steps, 'trace_environment_ids': list(ids),
                   'trace_layout': self.layout, 'physics_history': [{k:(v[ids].detach().cpu() if isinstance(v,torch.Tensor) else v) for k,v in row.items()} for row in self.history],
                   'physics': {k: v[ids].detach().cpu() for k, v in self.views.items()},
                   'learner_state': self.runner._checkpoint_state(None),
                   'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(),
                           'numpy': np.random.get_state(), 'python': random.getstate()},
                   'ledger_before': self.ledger.state()}
        if observations is not None:
            payload['observations'] = {str(k): v[ids].detach().cpu() for k, v in observations.items(include_nested=True, leaves_only=True)}
        if rewards is not None: payload['rewards'] = rewards[ids].detach().cpu()
        if sources is not None: payload['sources'] = {k: v[ids].detach().cpu() for k, v in sources.items()}
        solver = self.manager._solver
        payload['model'] = {}
        for key in ['body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'dof_armature', 'dof_invweight0', 'jnt_range', 'jnt_solref', 'geom_friction']:
            try:
                value = self.wp.to_torch(getattr(solver.mjw_model, key))
                payload['model'][key] = value[ids if value.shape[0] > max(ids) else [0]].detach().cpu()
            except Exception as error: payload['model'][key] = str(error)
        try:
            contact = solver.mjw_data.contact
            worlds = self.wp.to_torch(contact.worldid)
            selected = torch.zeros_like(worlds, dtype=torch.bool)
            for i in ids: selected |= worlds == i
            indices = selected.nonzero().flatten()[:256]
            payload['contacts'] = {}
            for name in ['worldid', 'dist', 'pos', 'frame', 'geom', 'dim', 'friction', 'solref', 'solimp']:
                try: payload['contacts'][name] = self.wp.to_torch(getattr(contact, name))[indices].detach().cpu()
                except Exception as error: payload['contacts'][name] = str(error)
        except Exception as error: payload['contact_capture_error'] = str(error)
        path = self.out/f'incident_{self.current_step}_{reason}.pt'
        torch.save(payload, path)
        self.event('incident_saved', ids=ids, reason=reason, file=str(path))
        return path

    def anomaly(self, ids, observations, rewards, sources):
        path = self.save_incident(ids, 'nonfinite', observations, rewards, sources)
        try: self.ledger.begin(self.current_step, ids)
        except RecoveryStop as error:
            self.event('recovery_rejected', ids=ids, reason=str(error), file=str(path))
            raise
        self.event('recovery_requested', ids=ids, file=str(path))

    def finish_step(self, result):
        values = [v for _, v in result[0].items(include_nested=True, leaves_only=True)]
        fast_finite([values, result[1]])
        pending = self.ledger.pending
        if pending and self.current_step > pending['step']:
            if not pending['physics_confirmed'] or self.reset_manager._has_pending_hard_reset:
                raise RecoveryStop('Recovery did not reach a verified physics step')
            ids = pending['ids']
            self.ledger.recovered()
            self.event('recovery_confirmed', ids=ids)

    def summary(self):
        return {**self.ledger.state(), 'physics_steps': self.physics_steps,
                'optimizer_steps_checked': self.optimizer_steps_checked,
                'student_optimizer_steps_checked': self.student_optimizer_steps_checked}

    def check_resources(self):
        import shutil
        gib = 1024**3
        free, total = self.torch.cuda.mem_get_info()
        current = int(Path('/sys/fs/cgroup/memory.current').read_text())
        limit = int(Path('/sys/fs/cgroup/memory.max').read_text())
        disk = shutil.disk_usage(self.out).free
        resources = {'gpu_free_bytes': free, 'memory_free_bytes': limit-current, 'disk_free_bytes': disk}
        if free < 2*gib or limit-current < 5*gib or disk < 10*gib:
            self.event('resource_headroom_exhausted', **resources)
            raise RuntimeError('Required GPU/RAM/disk headroom exhausted: '+str(resources))
        return resources
