#!/usr/bin/env python3
"""Run selected independently sourced Landsat dates against the unmodified
UrbanRay v3 88-building physics model, preserving separated dev/heldout scenes.

Run in Colab from inside /content/UrbanRay_Latest_Complete after uploading the
original Landsat qualified pixel CSV and the provided 10-date plan CSV.

Example (two-date smoke test):
python UrbanRay_Run_10Date_Validation.py --dates 2025-04-23 2026-05-12

Complete 10-date experiment:
python UrbanRay_Run_10Date_Validation.py --all

WARNING: Potentially computationally expensive (especially 16 rays x 10 days).
This script requires network for real Open-Meteo archives; never invents weather.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DEFAULT_PLAN='UrbanRay_10Date_Validation_Plan.csv'
DEFAULT_PIXELS='urbanray_qualified_landsat_pixels_2025_2026.csv'

def load_csv(path):
    with open(path,newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames:raise ValueError('Empty CSV: '+str(path))
        return reader.fieldnames,list(reader)

def run(*args):
    print('\n$',' '.join(map(str,args)),flush=True)
    subprocess.run([sys.executable,*map(str,args)],cwd=str(ROOT),check=True)

def provenance_ok(weather):
    p=weather.with_suffix('.provenance.json')
    if not weather.exists() or not p.exists():return False
    prov=json.loads(p.read_text())
    return (prov.get('csv_sha256')==hashlib.sha256(weather.read_bytes()).hexdigest()
            and 'archive-api.open-meteo.com' in prov.get('source_url',''))

def audit_valid(audit_path,rays):
    if not audit_path.exists():return False
    audit=json.loads(audit_path.read_text())
    if (not audit.get('weather_provenance_verified')
       or audit.get('core_buildings')!=88 or audit.get('context_included')
       or audit.get('warmup_hours')!=48 or audit.get('evaluated_hours')!=24
       or audit.get('operator',{}).get('rays_per_surface')!=rays):
        raise ValueError('Existing output is inconsistent or unverified: '+str(audit_path))
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',default=DEFAULT_PLAN)
    ap.add_argument('--pixels',default=DEFAULT_PIXELS)
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--dates',nargs='+',help='Run chosen dates, e.g. 2025-04-23 2026-05-12')
    g.add_argument('--all',action='store_true',help='Run all ten planned dates')
    ap.add_argument('--rays',type=int,default=16)
    ap.add_argument('--dry-run',action='store_true',help='Print planned date commands without downloading or simulating')
    a=ap.parse_args()
    if a.rays<1:raise ValueError('rays must be >= 1')
    if not (ROOT/'run_physics_plus.py').exists():
        raise FileNotFoundError('Put this script in the extracted UrbanRay_Validation_v3 project folder')
    plan_path=ROOT/a.plan
    pixels_path=ROOT/a.pixels
    for path in (plan_path,pixels_path):
        if not path.exists():raise FileNotFoundError('Missing required CSV: '+str(path))
    _,plan=load_csv(plan_path)
    pix_cols,pixels=load_csv(pixels_path)
    for col in ('scene_id','acquisition_utc','lst_c','projection_affine','st_uncertainty_k'):
        if col not in pix_cols:raise ValueError('Missing pixel field: '+col)
    plan_by_date={r['date_utc']:r for r in plan}
    if len(plan_by_date)!=len(plan):raise ValueError('Plan has duplicate dates')
    dates=list(plan_by_date) if a.all else a.dates
    if len(dates)!=len(set(dates)):raise ValueError('Duplicate requested date')
    for date in dates:
        if date not in plan_by_date:raise ValueError('Date not in pre-registered plan: '+date)
    print('Selected:',dates,'\nSplit:',{d:plan_by_date[d]['split'] for d in dates})
    if a.dry_run:
        for date in dates:
            r=plan_by_date[date]
            print('\n',date, r['split'],r['qualified_30m_grid_cells'],'screened grid cells')
            print(' ',r['weather_command'])
            print(' ',r['simulation_command'])
            print(' ',r['validation_command'])
        return
    output=ROOT/'outputs'/'multidate_validation'
    output.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data';data.mkdir(exist_ok=True)
    all_scenes=[]
    baseline_hash=None
    for index,date in enumerate(dates,1):
        print('\n===== DATE',index,'/',len(dates),date,'=====',flush=True)
        row=plan_by_date[date];scene_id=row['scene_id']
        observations=[r for r in pixels if r['scene_id']==scene_id]
        expected=int(row['qualified_30m_grid_cells'])
        if len(observations)!=expected:raise ValueError(f'{date}: expected {expected} grid rows, found {len(observations)}')
        if any(r['acquisition_utc']!=row['acquisition_utc'] for r in observations):
            raise ValueError('Inconsistent scene acquisition timestamps')
        obs_csv=data/f'landsat_obs_{date}.csv'
        with open(obs_csv,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=pix_cols);writer.writeheader();writer.writerows(observations)
        wstart=row['weather_start_utc'];wend=row['weather_end_utc']
        if (datetime.date.fromisoformat(wend)-datetime.date.fromisoformat(wstart)).days!=2:
            raise ValueError('Expected exactly two preceding warm-up days')
        weather=data/f'weather_{date}.csv'
        if not provenance_ok(weather):
            if weather.exists() or weather.with_suffix('.provenance.json').exists():
                raise ValueError('Existing weather lacks valid archive provenance: '+str(weather))
            run('fetch_weather_range.py','--start',wstart,'--end',wend,'--out',weather)
        model=ROOT/'outputs'/f'landsat_{date}'
        faces=model/'physics_training_faces.csv'
        audit=model/'audit.json'
        if not faces.exists() or not audit_valid(audit,a.rays):
            if model.exists() and list(model.iterdir()):
                raise ValueError('Partial/inconsistent model output exists. Examine before rerunning: '+str(model))
            run('run_physics_plus.py','--weather-csv',weather,'--output',model,
                '--warmup-hours','48','--hours','24','--rays',str(a.rays))
        if not audit_valid(audit,a.rays):raise ValueError('Physics audit failed: '+str(model))
        ah=json.loads(audit.read_text())
        if baseline_hash is None:baseline_hash=ah['geometry_sha256']
        elif baseline_hash!=ah['geometry_sha256']:raise ValueError('Geometry changed between dates')
        val_dir=ROOT/'outputs'/f'validation_{date}'
        run('validate_landsat_pixels.py','--pixels',obs_csv,'--faces',faces,
            '--max-st-uncertainty','5.0','--min-cloud-distance','0.0','--out',val_dir)
        scene_csv=val_dir/'scene_level_EXPLORATORY.csv'
        _,records=load_csv(scene_csv)
        for record in records:
            if record['scene_id']!=scene_id:raise ValueError('Wrong scene found in validated output')
            record['split_preregistered']=row['split']
            record['date_utc']=date
            all_scenes.append(record)
        print('Matched scenes for',date,':',len(records),flush=True)
    combined=output/'scene_level_EXPLORATORY.csv'
    if all_scenes:
        fieldnames=list(all_scenes[0])
        with open(combined,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader();writer.writerows(all_scenes)
        unique_dates=len(set(r['date_utc'] for r in all_scenes))
        print('\nCombined',len(all_scenes),'observed scenes from',unique_dates,'different dates:',combined)
        print('The scene-level diagnostic is NOT independently validated roof-temperature accuracy.')
        # Do not silently move reserved test scenes into the calibration group
        # when some development scenes fail spatial/time matching.
        found_dev={r['date_utc'] for r in all_scenes if r['split_preregistered']=='development'}
        found_test={r['date_utc'] for r in all_scenes if r['split_preregistered']=='heldout'}
        planned_dev={r['date_utc'] for r in plan if r['split']=='development'}
        planned_test={r['date_utc'] for r in plan if r['split']=='heldout'}
        if (found_dev==planned_dev and found_test==planned_test
                and len(found_dev)>=3 and len(found_test)>=3):
            run('evaluate_scene_holdout.py','--scenes',combined,'--out',output/'heldout')
        else:
            print('Holdout evaluation skipped: pre-registered development/test groups are not both complete.')
    else:
        print('No scenes matched: inspect individual date VALIDATION_STATUS.json files.')
    print('Finished. Saved output under:',output)

if __name__=='__main__':main()
