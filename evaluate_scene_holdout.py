"""Optional chronologically held-out *scene-level diagnostic*, never roof validation.

Requires >=6 independent acquisition dates, >=3 before/after split. Uses only
an additive bias fitted to earlier scenes as a cautious diagnostic and reports
both raw and corrected errors on untouched later dates. No model hyperparameter
optimization or temperature-physics validation is implied.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def score(y,yp):
    e=np.asarray(yp,dtype=float)-np.asarray(y,dtype=float)
    return {'MAE_C':float(np.abs(e).mean()),'RMSE_C':float(np.sqrt(np.mean(e**2))),
            'mean_bias_C':float(e.mean())}

def evaluate(csv,out):
    s=pd.read_csv(csv)
    need={'scene_id','acquisition_utc','observed_scene_mean_C','modeled_scene_mean_proxy_C'}
    if not need.issubset(s):raise ValueError(f'Missing columns {need-set(s)}')
    s['acquisition_utc']=pd.to_datetime(s['acquisition_utc'],utc=True,errors='raise')
    s['date_utc']=s.acquisition_utc.dt.date.astype(str)
    s=s.sort_values('acquisition_utc').drop_duplicates('scene_id')
    days=sorted(s.date_utc.unique())
    if len(days)<6:
        raise ValueError(f'Need >=6 distinct acquisition dates for held-out study; only {len(days)}. No score reported.')
    split=max(3,int(np.ceil(len(days)*.5)))
    if len(days)-split<3:raise ValueError('Insufficient untouched test dates')
    train=s[s.date_utc.isin(days[:split])]
    test=s[s.date_utc.isin(days[split:])].copy()
    correction=float((train.observed_scene_mean_C-train.modeled_scene_mean_proxy_C).mean())
    test['bias_corrected_model_C']=test.modeled_scene_mean_proxy_C+correction
    result={'status':'EXPLORATORY_SCENE_LEVEL_DIAGNOSTIC_NOT_REAL_ROOF_VALIDATION',
            'calibration_dates':days[:split],'heldout_dates':days[split:],
            'calibration_scene_count':len(train),'test_scene_count':len(test),
            'calibration_additive_bias_C':correction,
            'heldout_raw':score(test.observed_scene_mean_C,test.modeled_scene_mean_proxy_C),
            'heldout_bias_corrected':score(test.observed_scene_mean_C,test.bias_corrected_model_C),
            'warning':'These are Landsat grid retrieval vs simplified emission proxy residuals, not certified surface-temperature accuracy; thermal pixels remain spatially correlated.'}
    path=Path(out);path.mkdir(parents=True,exist_ok=True)
    test.to_csv(path/'heldout_scenes_EXPLORATORY.csv',index=False)
    (path/'heldout_diagnostic.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--scenes',required=True,help='scene_level_EXPLORATORY.csv from actual Landsat matching')
    p.add_argument('--out',default='outputs/validation_v3/heldout')
    a=p.parse_args()
    print(json.dumps(evaluate(a.scenes,a.out),indent=2))
