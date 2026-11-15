"""Plan reproducible 48h-warmup + 24h evaluation runs for actual Landsat scene dates."""
from __future__ import annotations
import argparse, json
from datetime import timedelta
from pathlib import Path
import pandas as pd

def plan(csv, output, existing_start=None, existing_end=None, min_pixels=3):
    scenes=pd.read_csv(csv)
    needed={'scene_id','acquisition_utc','qualified_30m_grid_cells'}
    if not needed.issubset(scenes.columns):
        raise ValueError(f'Missing scene fields: {sorted(needed-set(scenes.columns))}')
    scenes['acquisition_utc']=pd.to_datetime(scenes['acquisition_utc'],utc=True,errors='coerce')
    scenes['qualified_30m_grid_cells']=pd.to_numeric(scenes['qualified_30m_grid_cells'],errors='coerce').fillna(0)
    scenes=scenes[(scenes['acquisition_utc'].notna()) &
                  (scenes['qualified_30m_grid_cells']>=min_pixels)].copy()
    scenes=scenes.sort_values('acquisition_utc').drop_duplicates('scene_id')
    rows=[]
    for _,s in scenes.iterrows():
        d=s.acquisition_utc.date()
        start=(d-timedelta(days=2)).isoformat();end=d.isoformat()
        within=bool(existing_start and existing_end and
                    pd.Timestamp(existing_start,tz='UTC')<=s.acquisition_utc<=
                    pd.Timestamp(existing_end,tz='UTC'))
        rows.append({'scene_id':s.scene_id,'acquisition_utc':str(s.acquisition_utc),
                     'date_utc':d.isoformat(),'qualified_30m_grid_cells':int(s.qualified_30m_grid_cells),
                     'already_within_model_dates':within,
                     'weather_start_utc_date':start,'weather_end_utc_date':end,
                     'weather_command':f'python fetch_weather_range.py --start {start} --end {end} --out data/weather_{d}.csv',
                     'simulation_command':f'python run_physics_plus.py --weather-csv data/weather_{d}.csv --output outputs/landsat_{d} --warmup-hours 48 --hours 24 --rays 16'})
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows,columns=['scene_id','acquisition_utc','date_utc','qualified_30m_grid_cells',
       'already_within_model_dates','weather_start_utc_date','weather_end_utc_date',
       'weather_command','simulation_command']).to_csv(out,index=False)
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--scenes',required=True)
    ap.add_argument('--out',default='outputs/validation_v3/scene_run_plan.csv')
    ap.add_argument('--min-pixels',type=int,default=3)
    a=ap.parse_args()
    rows=plan(a.scenes,a.out,'2025-06-20','2025-06-23T23:00:00',a.min_pixels)
    print(json.dumps({'usable_scene_candidates':len(rows),'run_plan_csv':a.out,
     'dates_needing_new_simulation':sorted({r['date_utc'] for r in rows if not r['already_within_model_dates']})},indent=2))
