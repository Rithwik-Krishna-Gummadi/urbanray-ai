"""Conservative Landsat-vs-UrbanRay *diagnostic*.

USGS Landsat 8/9 Level-2 ST comes from ~100m TIRS observations, resampled
on a 30m grid. This script intersects exact *modeled* roof/ground triangles
with approximate 30m output-grid cells, and compares a fourth-power surface
emission proxy at the overpass time. Neither per-pixel MAE nor this proxy is
independent real roof-thermometry accuracy; use scene-level descriptive stats.

Input: export_landsat_validation_v3.js output CSV, and the user's actual
multi-day UrbanRay results ZIP (or the face CSV). No synthetic filling allowed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from integrated88 import load_verified_geometry
from urbanray import build_scene,CORE_BBOX

REQUIRED_PIXELS={
 'scene_id','acquisition_utc','projection_crs','projection_affine',
 'center_lon','center_lat','lst_c','st_uncertainty_k',
 'cloud_distance_km','qa_pixel','qa_radsat'
}
REQUIRED_FACES={'face_id','scenario','time_utc','surface_temp_C_simulated',
                'area_m2','x_m','y_m','kind','air_temp_C_grid'}
MODEL_CRS='EPSG:32618'


def readable_timestamp(s):
    t=pd.to_datetime(s,utc=True,errors='coerce')
    if pd.isna(t): raise ValueError(f'Invalid UTC timestamp: {s}')
    return t


def quality_reason(row,max_uncertainty=3.0,min_cloud_distance=0.5):
    """Defense in depth: independently check masks applied in EE."""
    try:
        if not np.isfinite(float(row['lst_c'])): return 'missing_temperature'
        if not (-90<=float(row['lst_c'])<=100): return 'out_of_range_temperature'
        if int(float(row['qa_pixel']))&191: return 'qa_pixel_bad'
        if int(float(row['qa_radsat']))!=0: return 'saturation_or_occlusion'
        if not (0<float(row['st_uncertainty_k'])<=max_uncertainty): return 'st_uncertainty_too_high'
        if float(row['cloud_distance_km'])<min_cloud_distance: return 'near_cloud'
        if not (-180<float(row['center_lon'])<180 and -90<float(row['center_lat'])<90): return 'invalid_coords'
        if not isinstance(row['projection_crs'],str) or not row['projection_crs'].strip(): return 'missing_crs'
        vals=[float(x) for x in str(row['projection_affine']).split(';')]
        if len(vals)!=6 or not np.isfinite(vals).all(): return 'invalid_pixel_affine'
    except (ValueError,TypeError,KeyError):
        return 'invalid_schema_value'
    return None


def pixel_polygon(row):
    """30m grid-cell footprint using actual Landsat affine basis vectors.

    Pixel's center comes from image pixelLonLat sampled in native ST_B10 grid;
    2D affine basis determines its orientation and dimensions. UTM transforms
    are locally exact to within projection sampling precision.
    """
    affine=np.array([float(x) for x in str(row['projection_affine']).split(';')])
    a,b,_c,d,e,_f=affine
    crs=str(row['projection_crs'])
    to_source=Transformer.from_crs('EPSG:4326',crs,always_xy=True)
    to_model=Transformer.from_crs(crs,MODEL_CRS,always_xy=True)
    cx,cy=to_source.transform(float(row['center_lon']),float(row['center_lat']))
    corners=[]
    for u,v in [(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)]:
        corners.append(to_model.transform(cx+u*a+v*b,cy+u*d+v*e))
    polygon=Polygon(corners)
    if not polygon.is_valid or polygon.area<500 or polygon.area>1500:
        raise ValueError('Unexpected satellite grid footprint, area, or projection')
    return polygon


def reconstructed_scene():
    # This EXACT call matches run_physics_plus.execute (core geometry, 25m ground).
    records,sha=load_verified_geometry(ROOT/'data'/'nyc_88_source.geojson')
    if len(records)!=88:raise AssertionError('Unexpected number of verified NYC buildings')
    scene=build_scene(records,ground_cell_m=25,max_buildings=len(records))
    if scene['source_is_synthetic']:raise AssertionError('Synthetic geometry not admissible')
    xmin,ymin,_,_=CORE_BBOX
    tx=Transformer.from_crs('EPSG:4326',MODEL_CRS,always_xy=True)
    ox,oy=tx.transform(xmin,ymin)
    def make_local_triangle(i):
        xyz=scene['triangles'][i]
        return Polygon([(float(x+ox),float(y+oy)) for x,y,z in xyz])
    kinds=np.asarray(scene['kind'])
    indices=np.where(np.isin(kinds,['roof','ground']))[0]
    top_polys=[];top_ids=[]
    for i in indices:
        poly=make_local_triangle(i)
        if poly.is_valid and poly.area>1e-8:
            top_polys.append(poly);top_ids.append(int(i))
    from shapely.geometry import box
    lonlat_core=box(*CORE_BBOX)
    core=shapely_transform(tx.transform,lonlat_core)
    return scene,sha,core,top_polys,np.array(top_ids,dtype=np.int32)


def load_faces(faces_csv=None,results_zip=None,member='historical_ml/physics_training_faces.csv'):
    if bool(faces_csv)==bool(results_zip):
        raise ValueError('Specify one of --faces or --results-zip')
    columns=list(REQUIRED_FACES|{'date_utc'})
    if results_zip:
        with zipfile.ZipFile(results_zip) as z:
            if member not in z.namelist():raise FileNotFoundError(f'Cannot find {member} in ZIP')
            with z.open(member) as f: df=pd.read_csv(f,usecols=lambda name:name in columns)
            audit_member=member.rsplit('/',1)[0]+'/audit.json'
            audit=json.loads(z.read(audit_member)) if audit_member in z.namelist() else None
    else:
        df=pd.read_csv(faces_csv,usecols=lambda name:name in columns)
        path=Path(faces_csv).parent/'audit.json'
        audit=json.loads(path.read_text()) if path.is_file() else None
    if not REQUIRED_FACES.issubset(df.columns):
        raise ValueError(f'Model output missing columns: {REQUIRED_FACES-set(df.columns)}')
    df=df[df.scenario=='dark_roof'].copy()
    if df.empty:raise ValueError('Only nominal dark_roof scenario is eligible for observational comparison')
    df['time_utc']=pd.to_datetime(df.time_utc,utc=True,errors='raise')
    df['face_id']=pd.to_numeric(df.face_id,errors='raise').astype(int)
    if df.duplicated(['time_utc','face_id']).any():raise ValueError('Duplicate face/time values')
    return df,audit


def interpolate_faces(faces,timestamp,scene, max_gap_minutes=65):
    """Interpolate an adjacent pair of complete hourly model states."""
    t=readable_timestamp(timestamp)
    times=sorted(faces.time_utc.unique())
    if len(times)==0:return None,'no_model_times'
    earlier=[x for x in times if x<=t]
    later=[x for x in times if x>=t]
    if not earlier or not later:return None,'time_outside_model_range'
    t0=earlier[-1];t1=later[0]
    if (t1-t0).total_seconds()/60>max_gap_minutes:
        return None,'gap_in_model_hourly_data'
    if max(abs((t-t0).total_seconds()),abs((t1-t).total_seconds()))/60>max_gap_minutes:
        return None,'model_time_too_far'
    f0=faces[faces.time_utc==t0].sort_values('face_id')
    if len(f0)!=len(scene['area']) or not np.array_equal(f0.face_id.to_numpy(),np.arange(len(scene['area']))):
        raise AssertionError('Model faces do not match the source 88-building scene')
    if not np.allclose(f0.area_m2.to_numpy(),scene['area'],rtol=1e-6,atol=1e-6):
        raise AssertionError('Model area alignment changed: wrong geometry or output')
    if not np.allclose(f0.x_m.to_numpy(),scene['center'][:,0],atol=1e-5):
        raise AssertionError('Model centroid alignment changed')
    if not np.allclose(f0.y_m.to_numpy(),scene['center'][:,1],atol=1e-5):
        raise AssertionError('Model centroid alignment changed')
    if t0==t1: return (f0.surface_temp_C_simulated.to_numpy(),
                       float(f0.air_temp_C_grid.iloc[0]),t0,t1),None
    f1=faces[faces.time_utc==t1].sort_values('face_id')
    if len(f1)!=len(f0) or not np.array_equal(f1.face_id.to_numpy(),f0.face_id.to_numpy()):
        raise AssertionError('Missing adjacent model faces')
    weight=(t-t0).total_seconds()/(t1-t0).total_seconds()
    temperatures=f0.surface_temp_C_simulated.to_numpy()*(1-weight)+f1.surface_temp_C_simulated.to_numpy()*weight
    air=float(f0.air_temp_C_grid.iloc[0]*(1-weight)+f1.air_temp_C_grid.iloc[0]*weight)
    return (temperatures,air,t0,t1),None


def radiance_proxy_C(temperatures_C,area_m2,emissivity):
    """Approximate bolometric emissivity-weighted temperature; not ST retrieval."""
    T=np.asarray(temperatures_C,dtype=float)+273.15
    area=np.asarray(area_m2,dtype=float)
    eps=np.asarray(emissivity,dtype=float)
    if np.any(area<=0) or np.any(eps<=0) or not np.isfinite(T).all():
        raise ValueError('Invalid temperature/area/emissivity')
    return float((np.sum(area*eps*T**4)/np.sum(area*eps))**0.25-273.15)


def pixel_estimate(poly,core,tree,top_ids,scene,temperatures,eps, min_in_core=0.98, min_coverage=0.90):
    area=poly.area
    inside=poly.intersection(core)
    core_frac=inside.area/area
    if core_frac<min_in_core:return None,'edge_of_core_or_external_thermal_support'
    # Exact 2D ground-plane intersection of modeled top-facing triangles.
    candidates=tree.query(poly,predicate='intersects')
    ids=[];weights=[]
    for local_i in candidates:
        fid=int(top_ids[int(local_i)])
        patch=tree.geometries[int(local_i)]
        overlap=patch.intersection(inside).area
        if overlap>1e-8:
            ids.append(fid);weights.append(overlap)
    if not weights:return None,'missing_modeled_visible_surfaces'
    total=float(sum(weights))
    coverage=total/area
    if coverage<min_coverage:return None,'insufficient_modeled_surface_coverage'
    if coverage>1.04:return None,'overlapping_roofs_or_geometry_ambiguity'
    ids=np.array(ids,dtype=int);w=np.array(weights,dtype=float)
    kind=np.asarray(scene['kind'])[ids]
    roof_frac=float(w[kind=='roof'].sum()/total)
    prediction=radiance_proxy_C(temperatures[ids],w,eps[ids])
    area_mean=float(np.average(temperatures[ids],weights=w))
    return {'modeled_emission_proxy_C':prediction,
            'modeled_area_mean_C':area_mean,
            'core_coverage_fraction':core_frac,
            'modeled_surface_coverage_fraction':coverage,
            'modeled_roof_fraction':roof_frac,
            'contributing_face_count':len(ids)},None


def run(pixels_csv,faces_csv=None,results_zip=None,member='historical_ml/physics_training_faces.csv',out='outputs/validation_v3',
        max_st_uncertainty=3.0,min_cloud_distance=0.5,min_in_core=.98,min_surface_coverage=.90,
        allow_synthetic_fixture=False):
    sat=pd.read_csv(pixels_csv)
    if not REQUIRED_PIXELS.issubset(sat.columns):
        raise ValueError(f'Missing Earth Engine pixel fields: {sorted(REQUIRED_PIXELS-set(sat.columns))}')
    if not allow_synthetic_fixture and (('SMOKE_ONLY' in str(pixels_csv).upper())
            or sat.scene_id.astype(str).str.contains('SMOKE_ONLY|FAKE|SYNTHETIC',case=False,regex=True).any()):
        raise ValueError('Synthetic fixture rejected: real-world validation needs actual Landsat export')
    scene,sha,core,polys,faceids=reconstructed_scene()
    faces,audit=load_faces(faces_csv,results_zip,member)
    if audit is not None:
        if audit.get('geometry_sha256')!=sha:
            raise AssertionError('Model results geometry SHA-256 does not match source 88 buildings')
        if audit.get('context_included'):
            raise ValueError('This matcher requires core-only geometry: buffered model uses different face IDs')
    tree=STRtree(polys)
    n=len(scene['area'])
    eps=np.array([{'roof':.91,'wall':.92,'ground':.95}[k] for k in scene['kind']])
    if audit and 'materials_assumed' in audit:
        eps=np.array([audit['materials_assumed'][k]['emissivity'] for k in scene['kind']])
    matched=[];rejected=[];pairs=[]
    for scene_id,g in sat.groupby('scene_id',dropna=False,sort=False):
        # Duplicate pixel center records within a scene are not independent observations.
        g=g.drop_duplicates(['center_lon','center_lat']).copy()
        times=g['acquisition_utc'].astype(str).unique()
        if len(times)!=1:
            rejected.append({'scene_id':scene_id,'reason':'inconsistent_acquisition_time_in_scene',
                             'rejected_rows':len(g)})
            continue
        interpolated,error=interpolate_faces(faces,times[0],scene)
        if error:
            rejected.append({'scene_id':scene_id,'reason':error,'rejected_rows':len(g)})
            continue
        temperatures,air,t0,t1=interpolated
        for _,r in g.iterrows():
            reason=quality_reason(r,max_st_uncertainty,min_cloud_distance)
            if reason:
                rejected.append({'scene_id':scene_id,'reason':reason,'rejected_rows':1});continue
            try:
                pixel=pixel_polygon(r)
                pred,reason=pixel_estimate(pixel,core,tree,faceids,scene,temperatures,eps,
                                          min_in_core,min_surface_coverage)
            except (ValueError,TypeError) as e:
                rejected.append({'scene_id':scene_id,'reason':'invalid_projection_or_geometry','rejected_rows':1})
                continue
            if reason:
                rejected.append({'scene_id':scene_id,'reason':reason,'rejected_rows':1});continue
            matched.append({'scene_id':str(scene_id),'acquisition_utc':str(times[0]),
                'center_lon':float(r.center_lon),'center_lat':float(r.center_lat),
                'observed_landsat_lst_C':float(r.lst_c),
                'landsat_uncertainty_K':float(r.st_uncertainty_k),
                'cloud_distance_km':float(r.cloud_distance_km),
                'retrieval_emissivity':float(r.get('retrieval_emissivity',np.nan)),
                'model_time_before_utc':t0.isoformat(),'model_time_after_utc':t1.isoformat(),
                'model_air_temperature_C':air,
                **pred,
                'difference_C_exploratory':float(pred['modeled_emission_proxy_C']-float(r.lst_c))})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    mdf=pd.DataFrame(matched,columns=['scene_id','acquisition_utc','center_lon','center_lat',
        'observed_landsat_lst_C','landsat_uncertainty_K','cloud_distance_km',
        'retrieval_emissivity','model_time_before_utc','model_time_after_utc',
        'model_air_temperature_C','modeled_emission_proxy_C','modeled_area_mean_C',
        'core_coverage_fraction','modeled_surface_coverage_fraction','modeled_roof_fraction',
        'contributing_face_count','difference_C_exploratory'])
    mdf.to_csv(out/'matched_pixels_EXPLORATORY.csv',index=False)
    rej=pd.DataFrame(rejected,columns=['scene_id','reason','rejected_rows'])
    rej.to_csv(out/'rejected_pixels_and_scenes.csv',index=False)
    if not mdf.empty:
        for (sid,t),g in mdf.groupby(['scene_id','acquisition_utc']):
            # 30m samples share 100m thermal data; aggregate to ONE scene-level record.
            pairs.append({'scene_id':sid,'acquisition_utc':t,'matched_30m_grid_cells':len(g),
                'observed_scene_mean_C':float(g.observed_landsat_lst_C.mean()),
                'modeled_scene_mean_proxy_C':float(g.modeled_emission_proxy_C.mean()),
                'mean_landsat_uncertainty_K':float(g.landsat_uncertainty_K.mean()),
                'mean_roof_fraction':float(g.modeled_roof_fraction.mean()),
                'scene_mean_difference_C_exploratory':float(g.difference_C_exploratory.mean()),
                'gridded_air_C_not_equivalent':float(g.model_air_temperature_C.mean())})
    sdf=pd.DataFrame(pairs,columns=['scene_id','acquisition_utc','matched_30m_grid_cells',
        'observed_scene_mean_C','modeled_scene_mean_proxy_C','mean_landsat_uncertainty_K',
        'mean_roof_fraction','scene_mean_difference_C_exploratory','gridded_air_C_not_equivalent'])
    sdf.to_csv(out/'scene_level_EXPLORATORY.csv',index=False)
    result={'status':'EXPLORATORY_DIAGNOSTIC_NOT_INDEPENDENT_ROOF_TEMPERATURE_VALIDATION',
       'source':'USGS Landsat Collection-2 Level-2 ST (if input came from specified EE export)',
       'source_geometry_sha256':sha,
       'source_satellite_file_sha256':hashlib.sha256(Path(pixels_csv).read_bytes()).hexdigest(),
       'source_model_weather_provenance_from_audit':None if not audit else audit.get('weather_source'),
       'source_model_weather_hash':None if not audit else audit.get('weather_sha256'),
       'usable_scene_count':len(sdf),'qualified_paired_30m_grid_cells':len(mdf),
       'rejected_rows_or_scenes':int(rej.rejected_rows.sum()) if len(rej) else 0,
       'method':'Landsat native 30m output grid, quality-masked, 30m 2D roof-ground overlap, time-interpolated surface emission proxy',
       'limitations':[
         'Native thermal observations are ~100m and resampled to 30m, so matched 30m cells are correlated, not independent.',
         'Model simulates homogeneous material-class prismatic faces, not measured emissivities or individual real rooftops.',
         'Modeled bolometric emissivity-weighted T^4 is not equivalent to Landsat spectral retrieval/atmospheric inversion.',
         'Temperature observations and geometry snapshot may refer to different years.',
         'Boundary mixing/thermal point spread function, vegetation, rooftops and façade visibility are inadequately resolved.',
         'The model weather hourly solar irradiance is a preceding-hour average, so a point-in-time overpass is only approximately matched.',
         'Reflective-roof intervention is hypothetical and cannot be validated by existing unmodified-roof satellite observations.'
        ]}
    if len(sdf):
        errors=sdf.scene_mean_difference_C_exploratory.to_numpy()
        result.update(scene_mean_diagnostic_MAE_C=float(np.abs(errors).mean()),
                      scene_mean_diagnostic_RMSE_C=float(np.sqrt(np.mean(errors**2))),
                      scene_mean_diagnostic_bias_C=float(errors.mean()))
        # Scenes are independent sampling units only approximately (some same date/track).
        result['distinct_acquisition_dates']=int(pd.to_datetime(sdf.acquisition_utc,utc=True).dt.date.nunique())
        result['independent_validation_claim_allowed']=False
        result['note']='Do not report these diagnostics as model temperature accuracy or calibrated error.'
    else:
        result['note']='NO VALID MATCHES; zero real-world accuracy measurements completed.'
    (out/'VALIDATION_STATUS.json').write_text(json.dumps(result,indent=2))
    if len(sdf):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(6.5,5))
            ax.scatter(sdf.observed_scene_mean_C,sdf.modeled_scene_mean_proxy_C)
            lo=min(sdf.observed_scene_mean_C.min(),sdf.modeled_scene_mean_proxy_C.min())
            hi=max(sdf.observed_scene_mean_C.max(),sdf.modeled_scene_mean_proxy_C.max())
            ax.plot([lo,hi],[lo,hi],linestyle='--',label='1:1 line')
            ax.set(xlabel='Landsat scene mean LST (°C)',ylabel='Modeled emission proxy (°C)',
                   title='EXPLORATORY scene-level comparison, not roof validation')
            ax.legend();fig.tight_layout();fig.savefig(out/'scene_diagnostic.png',dpi=160);plt.close(fig)
        except ImportError: pass
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pixels',required=True,help='Google Earth Engine Landsat export CSV')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--faces',help='physics_training_faces.csv')
    source.add_argument('--results-zip',help='original UrbanRay_PhysicsPlus_Results.zip')
    p.add_argument('--zip-member',default='historical_ml/physics_training_faces.csv')
    p.add_argument('--out',default='outputs/validation_v3')
    p.add_argument('--max-st-uncertainty',type=float,default=3.0)
    p.add_argument('--min-cloud-distance',type=float,default=0.5)
    p.add_argument('--allow-synthetic-fixture',action='store_true',help='Unit-test only; invalidates any observed-data claim')
    args=p.parse_args()
    report=run(args.pixels,faces_csv=args.faces,results_zip=args.results_zip,
               member=args.zip_member,out=args.out,max_st_uncertainty=args.max_st_uncertainty,
               min_cloud_distance=args.min_cloud_distance,
               allow_synthetic_fixture=args.allow_synthetic_fixture)
    print(json.dumps(report,indent=2))
