"""Generate bilingual delivery and learning curves after evaluation."""
import json
from pathlib import Path
from common import HERE,ROOT,atomic_json,sha,TARGET_ITERATIONS

def create_report(state,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    for folder in state['training_sessions']:
        rows.extend(json.loads(line) for line in (Path(folder)/'iterations.jsonl').read_text().splitlines())
    rows={r['iterations_completed']:r for r in rows}
    ordered=[rows[k] for k in sorted(rows)]
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    x=[r['iterations_completed'] for r in ordered]
    for name in sorted({k for r in ordered for k in r['losses']}):
        axes[0,0].plot(x,[r['losses'].get(name,float('nan')) for r in ordered],label=name,linewidth=.6)
    axes[0,0].set_yscale('symlog');axes[0,0].set_title('Losses');axes[0,0].legend(fontsize=6)
    axes[0,1].plot(x,[r['zero_command_time_fraction'] for r in ordered]);axes[0,1].set_title('Measured cumulative zero-command fraction')
    for group in ['teacher','student']:
        axes[1,0].plot(x,[r.get('group_metrics',{}).get(group,{}).get('terrain_level',float('nan')) for r in ordered],label=group,linewidth=.6)
    axes[1,0].set_title('Terrain level');axes[1,0].legend()
    for i in range(8):axes[1,1].plot(x,[r.get('expert_mean_usage',[float('nan')]*8)[i] for r in ordered],label=str(i),linewidth=.6)
    axes[1,1].set_title('Expert gate usage')
    for ax in axes.flat:ax.set_xlabel('Iteration');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(HERE/'training_curves.png',dpi=160);plt.close(fig)
    recommended=result['recommended_iteration']
    text=f'''# D2 MoE-CTS 混合地形长训 / Mixed-terrain training

8192 environments, Newton, seed 42, {TARGET_ITERATIONS} iterations, 24 steps per iteration.
6144 teacher / 2048 student environments. Fresh initialization; no old weights loaded.
纯零命令课程：0–1500 轮由 0 增至 10%，零分支随机转向关闭。
推荐模型 / Recommended iteration: **{recommended if recommended is not None else 'NONE / 无联合达标模型'}**.
最终检查点 / Final checkpoint: {state['checkpoint']}
评测错误数 / Evaluation errors: {result['error_count']}.

训练配置、源代码哈希及受保护模型记录：manifest.json / experiment.json。
容量与耗时：capacity_status.json。完整评测：evaluation_status.json。
曲线：training_curves.png。导出及视频：evaluation/model_<iteration>/。
本实验改变了算法和训练地形，不能把相对 the supplied baseline 的差异归因于单一因素。
The comparison changes both algorithm and training terrain; it is not a single-factor causal experiment.
No original models were replaced. No automatic extension beyond {TARGET_ITERATIONS} iterations.

键盘回放 / Keyboard playback:
# Activate your Isaac Lab Python environment first.
python deploy/deploy_mujoco/deploy.py --config deploy/deploy_mujoco/d2.yaml --policy {HERE}/evaluation/model_{recommended if recommended is not None else TARGET_ITERATIONS-1}/exported/policy.pt --terrain stairs --keyboard --duration 600

手动检查 / Manual status: cat {HERE}/status.json
复测 / Re-evaluate: python {HERE}/workflow.py --evaluate-only
'''
    comparisons=[]
    def compare(label,cases,stairs):
        rates=[v['success_rate'] if isinstance(v,dict) else sum(t['passed'] for t in v)/len(v) for v in stairs.values()]
        comparisons.append({'model':label,
            'drift_cm':cases.get('stand_d2',{}).get('max_stand_drift_m',float('nan'))*100,
            'speed_error':cases.get('forward_d2',{}).get('worst_relative_speed_error',float('nan')),
            'worst_stair_rate':min(rates) if rates else float('nan')})
    for label,base in result['baselines'].items():compare(label,base['flat'],base['stairs'])
    for iteration in sorted({TARGET_ITERATIONS-1,recommended}-{None}):
        candidate=result['candidates'][f'model_{iteration}']
        compare(f'moe_{iteration}',candidate.get('cases',{}),candidate.get('stairs',{}))
    text+='\n| Model | Stand drift cm (delay 2) | Forward speed error | Worst stair success |\n|---|---:|---:|---:|\n'
    for item in comparisons:
        text+=f"| {item['model']} | {item['drift_cm']:.3f} | {item['speed_error']:.2%} | {item['worst_stair_rate']:.0%} |\n"
    fig,axes=plt.subplots(1,2,figsize=(10,4));labels=[c['model'] for c in comparisons]
    axes[0].bar(labels,[c['drift_cm'] for c in comparisons]);axes[0].set_ylabel('120s stand drift (cm)')
    axes[1].bar(labels,[c['worst_stair_rate'] for c in comparisons]);axes[1].set_ylabel('Worst stair success rate');axes[1].set_ylim(0,1)
    fig.tight_layout();fig.savefig(HERE/'comparison.png',dpi=160);plt.close(fig)
    atomic_json(HERE/'comparison_summary.json',comparisons)
    (HERE/'REPORT.md').write_text(text)
    inventory={str(p.relative_to(HERE)): {'bytes':p.stat().st_size,'sha256':sha(p)} for p in HERE.rglob('*')
               if p.is_file() and p.suffix in {'.onnx','.mp4'} or p.is_file() and p.name in {'policy.pt','config.yaml','REPORT.md','training_curves.png'}}
    atomic_json(HERE/'delivery_inventory.json',inventory)
