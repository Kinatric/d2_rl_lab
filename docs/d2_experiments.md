# Reproducible D2 experiments

These scripts preserve the deployed MoE-CTS audit, recovery and student evaluation behavior. They do not ship trained D2 weights or results. Activate the Python environment described in the README and build/validate D2 assets first. Linux, NVIDIA CUDA and a finite cgroup-v2 memory limit are required by the resource monitor.

## Create and run

```bash
python scripts/d2_experiments/init_experiment.py --output tests/upgrade_isaaclab_3_ea/my_moe
E="$PWD/tests/upgrade_isaaclab_3_ea/my_moe"
python "$E/self_test_recovery.py"
python "$E/preflight.py"
python "$E/evaluation_smoke.py"
python "$E/workflow.py"
```

Run the last command inside tmux for SSH-independent execution. Do not run the templates in `scripts/d2_experiments` directly. Initialization refuses existing output directories and records source/config hashes. All generated evidence stays under ignored `tests/` and `logs/`.

The default is Newton, seed 42, 8192 environments, 24 steps per iteration and **180000 total iterations**, with 6144 teacher/2048 student environments. The experiment retains 8 experts and the zero-command probability ramp to 10% in 1500 iterations. Only the experiment disables random yaw in the zero branch. It does not change task defaults or rewards.

`experiment.example.json` fixes the supported configuration; only `target_iterations` (integer >= 501) is configurable in this workflow. Supply an edited copy with `--config`. An optional `--baseline-policy /path/to/exported/policy.pt` enables baseline comparison; its matching `config.yaml` must be present. Without it, comparison is skipped explicitly.

Preflight runs 32-environment rollout/update, save/resume and student export checks, plus isolated/repeated/multiple-world faults, main/student gradient faults and failed recovery. The evaluation smoke checks inference, scoring, stairs and video mechanics; an untrained policy need not pass walking acceptance.

Capacity trials use 1024/4096 environments for 10 iterations, then 8192 for 100. Optimization-only memory pressure permits increasing minibatches from 4 to 8/16/32. Initialization/sampling pressure does not. Fresh formal training starts only after capacity and preflight pass. Reserve thresholds are 2 GiB GPU, 5 GiB cgroup RAM and 10 GiB disk; no other jobs are stopped. Failures remain available for diagnosis rather than blind restart.

## Resume and inspect

```bash
cat "$E/status.json"
python "$E/workflow.py" --resume-checkpoint /path/to/this-experiment/model_13000.pt \
  --diagnosis "External instance restart; checkpoint integrity verified"
```

Use the same experiment directory. The checkpoint must belong to a recorded formal session. Restore checks compare model parameters, both Adam states (moments, counters and parameter groups), and policy identity exactly before rollout. Resume begins at saved iteration + 1; the total target is unchanged. Learning rates, curriculum step counters and anomaly ledger are restored. Physics state, per-environment terrain assignment and observation history are reinitialized; trajectory replay is not promised.

Each session has separate `audit.json`, `iterations.jsonl`, `resources.json`, `recovery_events.jsonl` and incident snapshots. Checkpoints are validated and atomically published every 500 iterations and at the final iteration. Both optimizers' gradients, parameters and states are checked for finite values. Isolated failures can reset; simultaneous failures, repeated failures within 100 iterations, the third incident in that window, failed recovery or learner non-finite values stop the run. This guard contains failures; it is not a root-cause fix for low-friction physics instability.

## Student evaluation

After training, the workflow screens every retained checkpoint and evaluates a shortlist plus the final model. It uses student TorchScript/ONNX, 120-second standing, continuous walk–stop–walk, 5/10/15 cm standard stairs with multiple delays/seeds, and diagnostic official stairs. Commands change without resetting history or physics. Joint acceptance is required for a recommendation; failed candidates do not lower thresholds. Inference equivalence and the 20 ms control budget are checked. Results, curves, videos, selection and a bilingual report remain private.

```bash
python "$E/workflow.py" --evaluate-only
python deploy/deploy_mujoco/deploy.py --config deploy/deploy_mujoco/d2.yaml \
  --policy /path/to/exported/policy.pt --keyboard --duration 600
```

Space clears velocity commands. Preserve the exported metadata beside each policy. Changing physics/model settings requires a new experiment and renewed validation.
