"""Real-NYC-geometry physics+ study. Not a verified urban-temperature forecast.

Run only with actually downloaded, consecutive Open-Meteo historical weather.
Outputs complete scientific provenance and optional per-face AI training corpus.
"""
from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
from shapely.geometry import shape
from dataclasses import replace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from integrated88 import load_verified_geometry,load_weather
from urbanray import build_scene
from physics_plus import (ASSUMED_MATERIALS,build_reciprocal_operator,materials_with_roof_alpha,
     sensitivity_materials,thermal_properties,solar_source_from_weather,shortwave_from_operator,
     step_two_layer)


def scenario_materials(use_sensitivity=False):
    base={'dark_roof':materials_with_roof_alpha(.18),
          'reflective_roof':materials_with_roof_alpha(.58)}
    if use_sensitivity:
        base.update({
          'capacity_minus20_pct':sensitivity_materials(.8),
          'capacity_plus20_pct':sensitivity_materials(1.2),
          'emissivity_minus003':sensitivity_materials(1.0,-.03),
          'emissivity_plus003':sensitivity_materials(1.0,+.03),
          'conductivity_minus20_pct':sensitivity_materials(conductivity_factor=.8),
          'conductivity_plus20_pct':sensitivity_materials(conductivity_factor=1.2)})
    return base


def execute(weather_csv,output_dir,rays=8,shadow_samples=4,warmup_hours=48,
            hours=24,seed=44,sensitivity=False,save_faces=True,bounces=3,context_geojson=None):
    src_geo=ROOT/'data'/'nyc_88_source.geojson'
    records,sha=load_verified_geometry(src_geo)
    core_ids={r['id'] for r in records}
    context_file_hash=None
    if context_geojson is not None:
        context_file=Path(context_geojson)
        context_file_hash=hashlib.sha256(context_file.read_bytes()).hexdigest()
        context=json.loads(context_file.read_text())
        if context.get('type')!='FeatureCollection':raise ValueError('Context must be a GeoJSON FeatureCollection')
        extras=[]; extra_ids=set()
        for f in context['features']:
            pr=f['properties'];bid=str(pr['id'])
            if bid in core_ids or bid in extra_ids:continue
            if 'NYC Open Data' not in pr.get('source',''):raise ValueError('Context geometry must carry official NYC provenance')
            h=float(pr['height_m']);feet=float(pr['height_roof_ft'])
            if not np.isclose(h,feet*.3048,atol=1e-4) or not 3<=h<=450:
                raise ValueError('Invalid NYC context height conversion')
            geom=shape(f['geometry'])
            if not geom.is_valid or geom.is_empty:raise ValueError('Invalid NYC context footprint')
            extras.append({'id':bid,'height_m':h,'geometry':geom,'source':'NYC OTI verified API'})
            extra_ids.add(bid)
        print('Additional official NYC context features:',len(extras),flush=True)
        records=records+extras
    scene=build_scene(records,ground_cell_m=25,max_buildings=len(records))
    scene_ids=set(scene['building_id'])-{'GROUND'}
    if scene['source_is_synthetic'] or not core_ids.issubset(scene_ids):
        raise AssertionError('A source NYC core building was lost from the model')
    core_faces=np.array([(b in core_ids or b=='GROUND') for b in scene['building_id']])
    if not core_faces.any():raise ValueError('Empty core evaluation area')
    weather=load_weather(weather_csv)
    count=warmup_hours+hours
    if len(weather)<count:raise ValueError(f'Need {count} consecutive weather hours, only {len(weather)} supplied')
    weather=weather.iloc[:count].copy()
    if not (weather['timestamp'].diff().dropna()==pd.Timedelta(hours=1)).all():
        raise ValueError('weather must be continuous hourly observations')
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    print(f'Building 88-building sparse view factors with {rays} rays per patch...',flush=True)
    P,sky,opmeta=build_reciprocal_operator(scene,rays=rays,seed=seed)
    n=len(scene['area'])
    kinds=scene['kind'];area=scene['area'];kinds_order=('roof','wall','ground')
    scenarios=scenario_materials(sensitivity)
    temp0=float(weather.iloc[0]['temperature_2m'])+273.15
    temp_s={name:np.full(n,temp0) for name in scenarios}
    temp_d={name:np.full(n,temp0) for name in scenarios}
    refK=float(weather['temperature_2m'].iloc[:max(1,warmup_hours)].mean())+273.15
    results=[];faces=[]
    for i,row in weather.iterrows():
        dt=row['timestamp'].to_pydatetime()
        if (i%6==0):print(f'  weather hour {i+1}/{count}',flush=True)
        src,sun=solar_source_from_weather(scene,dt,
            max(0.,float(row['direct_normal_irradiance_instant'])),
            max(0.,float(row['diffuse_radiation_instant'])),sky,
            shadow_samples=shadow_samples,seed=seed+int(i))
        for name,materials in scenarios.items():
            props=thermal_properties(kinds,materials)
            sw,swmeta=shortwave_from_operator(src,P,sky,area,props['alpha'],bounces=bounces)
            Ts,Td,thermalmeta=step_two_layer(temp_s[name],temp_d[name],sw,
                air_K=float(row['temperature_2m'])+273.15,
                wind_ms=float(row['wind_speed_10m']),
                rh_percent=float(row['relative_humidity_2m']),
                cloud_fraction=float(row['cloud_cover'])/100,
                P=P,sky=sky,scene=scene,materials=materials,
                deep_reference_K=refK,dt=3600)
            temp_s[name],temp_d[name]=Ts,Td
            if i < warmup_hours:continue
            metrics={k:float(np.average(Ts[(kinds==k)&core_faces]-273.15,
                        weights=area[(kinds==k)&core_faces])) for k in kinds_order}
            results.append({'time_utc':dt.isoformat(),'date_utc':dt.date().isoformat(),
                'hour_in_evaluation':int(i-warmup_hours),
                'scenario':name,'roof_alpha_assumed':materials['roof'].alpha,
                'roof_C_simulated':metrics['roof'],
                'wall_C_simulated':metrics['wall'],
                'ground_C_simulated':metrics['ground'],
                'total_shortwave_absorbed_MW':swmeta['absorbed_W']/1e6,
                'SW_relative_budget_residual':swmeta['relative_residual'],
                'SW_unresolved_fraction':swmeta['unresolved_W']/max(1,swmeta['initial_W']),
                'LW_net_MW':thermalmeta['LW_net_MW'],
                'max_thermal_surface_residual_Wm2':thermalmeta['max_surface_residual_Wm2'],
                'max_thermal_substrate_residual_Wm2':thermalmeta['max_deep_residual_Wm2'],
                'air_temp_C_grid':float(row['temperature_2m'])})
            if save_faces:
                face = pd.DataFrame({
                    'time_utc':dt.isoformat(),'date_utc':dt.date().isoformat(),
                    'scenario':name,'face_id':np.arange(n),
                    'building_id':scene['building_id'],'kind':kinds,
                    'x_m':scene['center'][:,0],'y_m':scene['center'][:,1],
                    'z_m':scene['center'][:,2],'nx':scene['normal'][:,0],
                    'ny':scene['normal'][:,1],'nz':scene['normal'][:,2],
                    'area_m2':area,'albedo_assumed':props['alpha'],
                    'sun_east':sun[0],'sun_north':sun[1],'sun_up':sun[2],
                    'DNI_grid_Wm2':float(row['direct_normal_irradiance_instant']),
                    'DHI_grid_Wm2':float(row['diffuse_radiation_instant']),
                    'air_temp_C_grid':float(row['temperature_2m']),
                    'wind_grid_ms':float(row['wind_speed_10m']),
                    'sky_fraction_modeled':sky,'source_flux_Wm2':src/area,
                    'direct_plus_sky_abs_Wm2':src/area*(1-props['alpha']),
                    'multibounce_abs_Wm2':sw,
                    'surface_temp_C_simulated':Ts-273.15})
                faces.append(face)
    table=pd.DataFrame(results)
    table.to_csv(output/'thermal_hourly_summary.csv',index=False)
    if save_faces and faces:
        pd.concat(faces,ignore_index=True).to_csv(output/'physics_training_faces.csv',index=False)
    weather_prov_path=Path(weather_csv).with_suffix('.provenance.json')
    verified_download=False
    if weather_prov_path.exists():
        provenance=json.loads(weather_prov_path.read_text())
        verified_download=(provenance.get('csv_sha256')==hashlib.sha256(Path(weather_csv).read_bytes()).hexdigest()
                           and 'archive-api.open-meteo.com' in provenance.get('source_url',''))
    if not verified_download:
        print('WARNING: weather file has no matching archived-API provenance hash; cannot attest sourced weather',flush=True)
    meta={'simulation':'88 real NYC footprint polygon prisms, approximate physics, assumed materials',
          'core_buildings':88,'context_included':bool(context_geojson),
          'total_buildings_simulated':int(scene['building_count']),
          'face_count':int(n),'geometry_sha256':sha,'context_geojson_sha256':context_file_hash,
          'geometry_historical_limit':'2026 snapshot; not guaranteed correct for each weather date',
          'weather_file':str(Path(weather_csv).name),
          'weather_source':'Open-Meteo archive file with matching provenance' if verified_download else 'UNVERIFIED input (may be synthetic); NOT claimable as observed weather',
          'weather_provenance_verified':bool(verified_download),
          'weather_sha256':hashlib.sha256(Path(weather_csv).read_bytes()).hexdigest(),
          'warmup_hours':int(warmup_hours),'evaluated_hours':int(hours),
          'sun_location_latitude':float(sum(scene['bbox'][i] for i in [1,3])/2),
          'sun_location_longitude':float(sum(scene['bbox'][i] for i in [0,2])/2),
          'operator':opmeta,'shadow_samples':shadow_samples,'bounces':bounces,
          'materials_assumed':{k:vars(v) for k,v in ASSUMED_MATERIALS.items()},
          'scenario_names':list(scenarios),
          'scenario_assumed_materials':{name:{k:vars(v) for k,v in mats.items()} for name,mats in scenarios.items()},
          'surface_temperature_accuracy':'NOT independently established',
          'water_evaporation_and_tree_canopies':'NOT included without verified geometry/material inputs'}
    (output/'audit.json').write_text(json.dumps(meta,indent=2))
    print('SAVED:',output, 'rows:',len(table),flush=True)
    return table,meta

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--weather-csv',required=True)
    p.add_argument('--output',default=str(ROOT/'outputs'/'physics_plus'))
    p.add_argument('--rays',type=int,default=8)
    p.add_argument('--shadow-samples',type=int,default=4)
    p.add_argument('--warmup-hours',type=int,default=48)
    p.add_argument('--hours',type=int,default=24)
    p.add_argument('--bounces',type=int,default=3)
    p.add_argument('--seed',type=int,default=44)
    p.add_argument('--sensitivity',action='store_true')
    p.add_argument('--no-face-csv',action='store_true')
    p.add_argument('--context-geojson',default=None)
    a=p.parse_args()
    execute(a.weather_csv,a.output,a.rays,a.shadow_samples,a.warmup_hours,a.hours,
            a.seed,a.sensitivity,not a.no_face_csv,a.bounces,a.context_geojson)
