"""UrbanRay 88-building integration: real NYC footprint geometry, modeled shortwave
radiation and transparent approximate thermal response. NO validated temperature claims.

Atmospheric observations are gridded reanalysis estimates, and material parameters
are declared assumptions. 2014 NYC DCP 3D surfaces are NOT silently mixed with the
newer NYC 88-building footprint snapshot.
"""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import pandas as pd
from shapely.geometry import shape
from urbanray import build_scene, simulate

SIGMA = 5.670374419e-8

@dataclass(frozen=True)
class Material:
    albedo: float
    emissivity: float
    conductivity: float        # W/(m K), ASSUMED
    volumetric_capacity: float # J/(m^3 K), ASSUMED

# These values are not measured for the NYC buildings.
MATERIALS = {
    'roof': Material(.18, .91, .85, 1.55e6),
    'wall': Material(.30, .92, 1.25, 1.80e6),
    'ground': Material(.12, .95, .76, 1.65e6),
}

def load_verified_geometry(path: str|Path):
    path=Path(path); data=path.read_bytes(); collection=json.loads(data)
    assert collection['type']=='FeatureCollection'
    records=[]
    seen=set()
    for f in collection['features']:
        p=f['properties']; geom=shape(f['geometry'])
        bid=str(p['id']); feet=float(p['height_roof_ft']); meters=float(p['height_m'])
        if bid in seen: raise ValueError('Duplicate building identifier: '+bid)
        seen.add(bid)
        if not np.isclose(meters,feet*.3048,rtol=0,atol=1e-5):raise ValueError('Height units inconsistency')
        if not geom.is_valid or geom.is_empty:raise ValueError('Invalid NYC polygon '+bid)
        if not (3<=meters<=450):raise ValueError('Implausible building height '+bid)
        records.append({'id':bid,'height_m':meters,'geometry':geom,'source':'NYC OTI footprint snapshot'})
    if len(records)!=88:raise ValueError(f'Expected 88 source buildings, got {len(records)}')
    return records,hashlib.sha256(data).hexdigest()

def load_weather(path):
    """Require source-aligned hourly gridded weather and reject missing fields.
    DNI and DHI are W/m2 instantaneous fields; wind is m/s (10-m standard).
    """
    df=pd.read_csv(path)
    required={'time_local','temperature_2m','relative_humidity_2m','wind_speed_10m',
              'cloud_cover','direct_normal_irradiance_instant','diffuse_radiation_instant'}
    if not required.issubset(df):raise ValueError('Weather data missing: '+str(sorted(required-set(df))))
    df['timestamp']=pd.to_datetime(df['time_local'],utc=True)
    if df['timestamp'].isna().any() or df['timestamp'].duplicated().any():raise ValueError('Invalid or duplicate weather timestamps')
    for col in required-{'time_local'}:
        df[col]=pd.to_numeric(df[col],errors='coerce')
        if df[col].isna().any():raise ValueError(f'Missing weather values for {col}')
    if not df['wind_speed_10m'].between(0,80).all():raise ValueError('Wind outside basic physical bounds')
    if not df['relative_humidity_2m'].between(0,100).all():raise ValueError('RH must be 0..100 percent')
    if not df['cloud_cover'].between(0,100).all():raise ValueError('Cloud cover must be 0..100 percent')
    if (df[['direct_normal_irradiance_instant','diffuse_radiation_instant']]<0).any().any():raise ValueError('Negative irradiance')
    return df.sort_values('timestamp').reset_index(drop=True)

def thermal_coefficients(kinds):
    """Diurnal penetration depth: approximation; not a measured construction assembly."""
    omega=2*np.pi/86400
    vals=[]
    for kind in kinds:
        mat=MATERIALS[kind]
        depth=np.sqrt(2*mat.conductivity/(mat.volumetric_capacity*omega))
        C=mat.volumetric_capacity*depth
        G=mat.conductivity/depth
        vals.append((C,G,mat.emissivity))
    return tuple(np.asarray(x) for x in zip(*vals))

def convection_h(surface_K,air_K,wind_10m,z_m,kind):
    """Pragmatic exterior forced/natural convection correlation with bounded local
    wind reduction. Not a street-canyon CFD wind solution."""
    from thermal_correlations import convection_coefficient
    z=np.maximum(1,np.asarray(z_m))
    # Log-law profile only meaningful above displacement height; below use bounded fraction.
    wind_local=float(wind_10m)*np.clip((z/10)**.18,.25,2.5)
    length=np.where(np.asarray(kind)=='wall',5.,3.)
    return np.maximum(1.,convection_coefficient(surface_K,air_K,wind_local,length,
                         vertical=np.asarray(kind)=='wall'))

def sky_emissivity(air_K,rh_percent,cloud_fraction):
    """Parameterization, not measurement of atmospheric long-wave flux."""
    from thermal_correlations import brunt_cloud_longwave
    return brunt_cloud_longwave(float(air_K),float(rh_percent)/100,float(cloud_fraction))

def thermal_step(temp_K,solar_Wm2,air_K,wind_ms,rh_percent,cloud_fraction,
                 sky_view,kinds,z_m,dt_seconds=3600):
    """Semi-implicit single-face thermal model, longwave linearized per iteration.
    Ambient and deep reference temperature are set to gridded 2m air temp;
    thermal boundary assumption, NOT observed subsurface temperatures.
    """
    temp=np.asarray(temp_K,dtype=float)
    solar=np.asarray(solar_Wm2,dtype=float)
    kinds=np.asarray(kinds)
    sky_view=np.clip(np.asarray(sky_view,dtype=float),0,1)
    C,G,eps=thermal_coefficients(kinds)
    Lsky,_=sky_emissivity(air_K,rh_percent,cloud_fraction)
    L_in=sky_view*Lsky+(1-sky_view)*SIGMA*air_K**4
    guess=temp.copy()
    # Fixed 4 iterations with physically stable linearized radiation and conduction.
    for _ in range(4):
        h=convection_h(guess,air_K,wind_ms,z_m,kinds)
        denom=C/dt_seconds+h+G+4*eps*SIGMA*guess**3
        numerator=(C/dt_seconds)*temp + solar + eps*L_in + h*air_K + G*air_K + 3*eps*SIGMA*guess**4
        guess=numerator/denom
    if not np.all(np.isfinite(guess)) or np.any((guess<180)|(guess>410)):
        raise ValueError('Thermal numerical instability or physically extreme temperature')
    return guess

def run(geometry_path,weather_csv,output_dir,*,rays=5,roof_albedos=(.18,.58),
        max_hours=24,seed=44,latitude=None,longitude=None,bounces=3):
    """2 albedo scenarios on exactly the same real geometry and historical weather.
    Outputs model-estimated surface temperatures only; no claim of measured accuracy.
    """
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    records,hash256=load_verified_geometry(geometry_path)
    weather=load_weather(weather_csv).iloc[:max_hours]
    if len(weather)==0:raise ValueError('No weather records')
    # Ensure chronological hourly steps (avoid passing disjoint snapshots to thermal storage).
    t=weather['timestamp'].sort_values()
    if len(t)>1 and not (t.diff().dropna()==pd.Timedelta(hours=1)).all():
        raise ValueError('Thermal time integration requires consecutive hourly weather')
    scene=build_scene(records,ground_cell_m=25,max_buildings=88)
    if scene['source_is_synthetic'] or scene['building_count']!=88:raise AssertionError('Wrong geometry')
    kind=scene['kind']; area=scene['area']; z=scene['center'][:,2]
    bbox=scene['bbox']
    actual_lat=(bbox[1]+bbox[3])/2
    actual_lon=(bbox[0]+bbox[2])/2
    sun_lat=actual_lat if latitude is None else float(latitude)
    sun_lon=actual_lon if longitude is None else float(longitude)
    solar_location_matches_geometry=(abs(sun_lat-actual_lat)<0.05 and abs(sun_lon-actual_lon)<0.05)
    if not solar_location_matches_geometry:
        print('WARNING: sun location differs from NYC geometry: counterfactual illumination, not actual NYC')
    results=[];face_snapshot=[]
    for alpha in roof_albedos:
        T=np.full(len(kind),float(weather.iloc[0]['temperature_2m'])+273.15)
        for i,row in weather.iterrows():
            dt=row['timestamp'].to_pydatetime()
            solar=simulate(scene,dt,dni=float(row['direct_normal_irradiance_instant']),
               dhi=float(row['diffuse_radiation_instant']),
               albedos={'roof':alpha,'wall':.30,'ground':.12},
               bounces=bounces,sky_rays=rays,bounce_rays=rays,direct_samples=max(2,rays//2),
               seed=seed+i,latitude=sun_lat,longitude=sun_lon)
            if solar['budget']['relative_error']>1e-9: raise AssertionError('SW energy imbalance')
            T=thermal_step(T,solar['absorbed_Wm2'],
                air_K=float(row['temperature_2m'])+273.15,
                wind_ms=float(row['wind_speed_10m']),
                rh_percent=float(row['relative_humidity_2m']),
                cloud_fraction=float(row['cloud_cover'])/100,
                sky_view=solar['sky_visible_fraction'],kinds=kind,z_m=z)
            vals={k:float(np.average((T-273.15)[kind==k],weights=area[kind==k]))
                 for k in ('roof','wall','ground')}
            results.append({'hour':i,'time_utc':dt.isoformat(),'roof_albedo_assumed':alpha,
                'roof_C_modeled':vals['roof'],'wall_C_modeled':vals['wall'],
                'ground_C_modeled':vals['ground'],
                'air_C_reanalysis':float(row['temperature_2m']),
                'SW_absorbed_MW':float(solar['absorbed_W'].sum())/1e6,
                'SW_budget_relative_residual':solar['budget']['relative_error'],
                'unresolved_reflected_pct':100*solar['budget']['unresolved_reflection_W']/max(1,solar['budget']['initial_W'])})
            if i==len(weather)-1:
                face_snapshot.append(pd.DataFrame({'surface_id':np.arange(len(kind)),
                    'building_id':scene['building_id'],'kind':kind,'area_m2':area,
                    'center_x_m':scene['center'][:,0],'center_y_m':scene['center'][:,1],
                    'center_z_m':z,'roof_albedo_scenario':alpha,
                    'shortwave_absorption_Wm2':solar['absorbed_Wm2'],
                    'surface_C_model_estimate':T-273.15}))
    df=pd.DataFrame(results);df.to_csv(out/'hourly_modeled_thermal.csv',index=False)
    pd.concat(face_snapshot).to_csv(out/'last_hour_surface_results.csv',index=False)
    metadata={'source':'Official NYC OTI footprints archived in prior project',
        'geometry_SHA256':hash256,'buildings':88,'surface_triangles':len(kind),
        'geometry_historical_date':'Snapshot archived 2026-11-14; buildings may differ from 2025',
        'city_planning_3d_model':'2014 survey used as optional geometry comparison ONLY; no unverified matching',
        'weather':'Open-Meteo gridded historical reanalysis; not ground station observations',
        'weather_csv':Path(weather_csv).name,'material_properties':'ASSUMED, not measured',
        'thermal_accuracy':'NOT independently validated',
        'local_wind':'approximate log/power-law proxy, no CFD',
        'reflection':'Lambertian Monte Carlo 3 bounces',
        'max_hours':len(weather),'rays_per_surface':rays,'bounces':bounces,
        'solar_latitude':sun_lat,'solar_longitude':sun_lon,
        'sun_location_matches_study_area':solar_location_matches_geometry,
        'sampling':{'sky':rays,'reflections_each_bounce':rays,'direct':max(2,rays//2)},
        'output':'modeled temperatures / absorbed shortwave, not observations'}
    (out/'audit_metadata.json').write_text(json.dumps(metadata,indent=2))
    return df,metadata
