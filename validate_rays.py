"""Ray-count convergence/repeatability: no claim that largest ray count is truth."""
import argparse,sys,json
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from integrated88 import load_verified_geometry
from urbanray import build_scene
from physics_plus import (build_reciprocal_operator,solar_source_from_weather,
                          shortwave_from_operator)


def run(rays_list,seeds,out):
    records,sha=load_verified_geometry(ROOT/'data'/'nyc_88_source.geojson')
    scene=build_scene(records,ground_cell_m=25,max_buildings=88)
    archive=json.loads((ROOT/'data'/'archived_irradiance_12_samples.json').read_text())
    source=next(x for x in archive if x['time_local'].startswith('2025-06-21T12:00:00'))
    dt=pd.Timestamp(source['time_local']).to_pydatetime()
    alpha=np.array([{'roof':.18,'wall':.30,'ground':.12}[k] for k in scene['kind']])
    rows=[];flux={}
    for rays in rays_list:
        for seed in seeds:
            P,sky,info=build_reciprocal_operator(scene,rays,seed)
            direct,sun=solar_source_from_weather(scene,dt,source['dni_Wm2'],source['dhi_Wm2'],
                                sky,shadow_samples=max(2,rays//2),seed=seed)
            SW,bal=shortwave_from_operator(direct,P,sky,scene['area'],alpha,bounces=3)
            face=SW*scene['area'];total=float(face.sum())
            rows.append({'rays':rays,'seed':seed,'total_absorbed_MW':total/1e6,
                'max_shortwave_relative_budget_residual':bal['relative_residual'],
                'unresolved_shortwave_fraction':bal['unresolved_W']/max(1,bal['initial_W']),
                'area_weighted_mean_absorbed_Wm2':float(np.average(SW,weights=scene['area']))})
            flux[(rays,seed)]=SW
            print('rays:',rays,'seed:',seed,'Absorbed MW:',round(total/1e6,4),flush=True)
    # Largest tested ray count is only a numerical reference, never ground truth.
    highest=max(rays_list)
    reference=np.mean([flux[(highest,seed)] for seed in seeds],axis=0)
    for row in rows:
        f=flux[(row['rays'],row['seed'])]
        row['patch_area_weighted_MAE_Wm2_vs_high_reference']=float(
            np.average(np.abs(f-reference),weights=scene['area']))
    table=pd.DataFrame(rows)
    agg=table.groupby('rays').agg(total_MW_mean=('total_absorbed_MW','mean'),
                                   total_MW_sd=('total_absorbed_MW','std'),
                                   replicates=('seed','count')).reset_index()
    agg['relative_std_percent']=100*agg.total_MW_sd/agg.total_MW_mean
    # Unlike a single 32-ray reference, this tracks seed-to-seed variation too.
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/'seed_ray_runs.csv',index=False)
    agg.to_csv(out/'convergence.csv',index=False)
    audit={'geometry_sha256':sha,'buildings':88,'ray_counts':rays_list,'seeds':seeds,
           'irradiance':'archived sample from previous project, NOT refetched this turn',
           'meaning':'Monte Carlo repeatability, not accuracy versus measured temperature',
           'source_datetime':source['time_local'], 'source_dni':source['dni_Wm2'],
           'source_dhi':source['dhi_Wm2']}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    print(agg.to_string(index=False));return agg

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--rays',nargs='+',type=int,default=[4,8,16,32])
    p.add_argument('--seeds',nargs='+',type=int,default=[44,76,112])
    p.add_argument('--out',default='outputs/ray_convergence')
    a=p.parse_args();run(a.rays,a.seeds,a.out)
