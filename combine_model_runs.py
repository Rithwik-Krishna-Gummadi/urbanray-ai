"""Combine several *real-geometry* per-day UrbanRay runs for Landsat matching.

Each run must have the exact same geometry hash and nominal material parameters.
The source weather provenance is checked, not fabricated or inferred.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import pandas as pd

def combine(runs,out):
    if len(runs)<1:raise ValueError('Specify at least one real model output directory')
    metadata=[];tables=[]
    for d in [Path(x) for x in runs]:
        audit_path=d/'audit.json';csv=d/'physics_training_faces.csv'
        if not audit_path.is_file() or not csv.is_file():
            raise FileNotFoundError(f'Need audit.json and physics_training_faces.csv in {d}')
        audit=json.loads(audit_path.read_text())
        if audit.get('core_buildings')!=88 or audit.get('context_included',True):
            raise ValueError('This matcher only supports the exact 88-building core-only scene')
        if not audit.get('weather_provenance_verified'):
            raise ValueError(f'Unverified weather input: {d}. Rejecting simulation as historical.')
        meta={k:audit.get(k) for k in ['geometry_sha256','core_buildings','face_count',
             'materials_assumed','operator','shadow_samples','bounces','context_included']}
        metadata.append((meta,audit))
        df=pd.read_csv(csv)
        df['run_dir']=d.name
        tables.append(df)
    first=metadata[0][0]
    for m,audit in metadata[1:]:
        for field in ['geometry_sha256','core_buildings','face_count','materials_assumed',
                      'context_included','operator','shadow_samples','bounces']:
            if m[field]!=first[field]:raise ValueError(f'Simulation mismatch: {field}')
    combined=pd.concat(tables,ignore_index=True)
    expected={'face_id','time_utc','scenario','surface_temp_C_simulated'}
    if not expected.issubset(combined):raise ValueError(f'Missing columns: {expected-set(combined)}')
    combined=combined.sort_values(['time_utc','scenario','face_id'])
    dup=combined.duplicated(['time_utc','scenario','face_id'])
    if dup.any():raise ValueError('Duplicate time/scenario/face rows: overlapping model runs cannot be combined')
    n=int(first['face_count']);counts=combined.groupby(['time_utc','scenario']).size()
    if not (counts==n).all():raise ValueError('Incomplete per-time face records')
    dest=Path(out);dest.parent.mkdir(parents=True,exist_ok=True)
    combined.drop(columns=['run_dir']).to_csv(dest,index=False)
    reference=metadata[0][1]
    result={**reference,'combined_runs':[str(x) for x in runs],
       'weather_source':'Multiple individually provenance-verified Open-Meteo archive sources',
       'weather_provenance_verified':True,
       'distinct_weather_sha256':[a.get('weather_sha256') for _,a in metadata],
       'combined_face_rows':int(len(combined)),
       'combined_unique_utc_hours':int(combined.time_utc.nunique()),
       'combined_csv_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    dest.with_name('audit.json').write_text(json.dumps(result,indent=2,default=str))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run_dirs',nargs='+')
    p.add_argument('--out',default='outputs/validation_v3/combined_faces/physics_training_faces.csv')
    a=p.parse_args()
    result=combine(a.run_dirs,a.out)
    print('Combined',result['combined_face_rows'],'rows and',result['combined_unique_utc_hours'],'distinct hours; output',a.out)
