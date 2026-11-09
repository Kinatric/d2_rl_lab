"""Strict long-training acceptance, evaluated from uninterrupted 50 Hz trajectories."""
import json
from pathlib import Path
import numpy as np


def phase_score(samples, start, end, command, metadata, restart=False):
    full=samples[(samples[:,0]>start+1e-7)&(samples[:,0]<=end+1e-7)]
    data=full[full[:,0]>=start+2.]
    assert len(data)>0
    desired=np.asarray(command);standing=not desired.any()
    velocity=data[:,[4,5,7]];error=velocity-desired
    rmse=np.sqrt(np.mean(error**2,axis=0));xy_rmse=float(np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1))))
    drift=np.linalg.norm(data[:,1:3]-data[0,1:3],axis=1)
    height=data[:,3];tilt=float(data[:,8].max());contact=float(np.mean(data[:,11]>5.))
    failures=[]
    def check(ok,label):
        if not ok:failures.append(label)
    check(np.isfinite(full).all(),'non_finite')
    check(metadata['first_fall_s'] is None or metadata['first_fall_s']>end,'fall')
    check(metadata['warnings']==0,'mujoco_warning')
    check(height.min()>.20,'min_height')
    check(tilt<20.,'tilt')
    check(contact<.01,'nonfoot_contact')
    active=np.abs(desired)>1e-9
    relative=float(np.max(np.abs(velocity.mean(axis=0)[active]-desired[active])/np.abs(desired[active]))) if active.any() else 0.
    quiet=full[full[:,0]>=start+5.]
    height_span=float(np.ptp(quiet[:,3]))
    latency=None
    if standing:
        check(drift.max()<=.10,'stand_drift')
        check(xy_rmse<=.01,'stand_xy_rmse')
        check(rmse[2]<=.05,'stand_yaw_rmse')
        check(.285<=height.mean()<=.345,'stand_mean_height')
        check(height_span<=.01,'stand_height_span_after_5s')
        check(tilt<10.,'stand_tilt')
    else:
        check(xy_rmse<.15,'movement_xy_rmse')
        check(rmse[2]<.25,'movement_yaw_rmse')
        check(relative<=.10+1e-12,'movement_mean_error')
        if restart:
            axis=int(np.argmax(np.abs(desired)))
            # Trailing 0.5 s average, evaluated at the END of each 25-sample window.
            smoothed=np.convolve(full[:,[4,5,7]][:,axis],np.ones(25)/25,mode='valid')
            times=full[24:,0]-start
            inside=np.abs(smoothed-desired[axis])<=.1*abs(desired[axis])+1e-12
            # 51 timestamps span a full second at 50 Hz (50 intervals).
            sustained=np.flatnonzero(np.convolve(inside.astype(int),np.ones(51,dtype=int),mode='valid')==51)
            if len(sustained):latency=float(times[sustained[0]])
            check(latency is not None and latency<=2.+1e-7,'restart_within_2s_and_hold_1s')
    return {'start_s':start,'end_s':end,'command':command,'standing':standing,'passed':not failures,
        'failures':failures,'mean_velocity':velocity.mean(axis=0).tolist(),'relative_speed_error':relative,
        'xy_rmse_m_s':xy_rmse,'rmse_vx_vy_yaw':rmse.tolist(),'settled_max_displacement_m':float(drift.max()),
        'mean_height_m':float(height.mean()),'height_span_after_5s_m':height_span,
        'min_height_m':float(height.min()),'max_tilt_deg':tilt,'nonfoot_contact_fraction':contact,
        'restart_latency_s':latency,'first_2s_displacement_m':float(np.linalg.norm(data[0,1:3]-full[0,1:3]))}


def score(folder):
    folder=Path(folder);metadata=json.loads((folder/'result.json').read_text())
    trajectory=folder/'trajectory.csv'
    if not trajectory.exists():trajectory=folder/'trajectory.csv.gz'
    samples=np.loadtxt(trajectory,delimiter=',',skiprows=1)
    assert samples.ndim==2 and len(samples)==round(metadata['seconds']*50)
    assert np.isfinite(samples).all() and metadata['finite'] and metadata['continuous_history']
    assert np.allclose(np.diff(samples[:,0]),.02,rtol=0,atol=1e-7)
    assert metadata['physics_steps']==round(metadata['seconds']/.005)
    phases=[phase_score(samples,p['start_s'],p['end_s'],p['command'],metadata,
                        restart=i>0 and not any(metadata['phases'][i-1]['command']) and any(p['command']))
            for i,p in enumerate(metadata['phases'])]
    result={'passed':all(p['passed'] for p in phases),'profile':metadata['profile'],
        'delay_steps':metadata['delay_steps'],'seconds':metadata['seconds'],'phases':phases,
        'worst_relative_speed_error':max(p['relative_speed_error'] for p in phases),
        'max_stand_drift_m':max(p['settled_max_displacement_m'] for p in phases if p['standing']),
        'first_fall_s':metadata['first_fall_s'],'finite':True,'continuous_history':True}
    if metadata['profile']=='stand' and metadata['seconds']>=120:
        result['first_30s']=phase_score(samples,0,30,[0,0,0],metadata)
    (folder/'strict_result.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    import sys
    print(json.dumps(score(sys.argv[1]),indent=2))
