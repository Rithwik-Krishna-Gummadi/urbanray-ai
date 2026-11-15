"""UrbanRay AI: reproducible geometry + approximate diffuse multi-bounce shortwave radiative transfer.

All radiation is W (per mesh face) or W/m2 (per face area); not a surface-temperature model.
NYC geometry and per-building roof heights are observational; materials are assumed.
"""
from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from shapely.geometry import shape, Polygon, MultiPolygon, box
from shapely.geometry.polygon import orient
from shapely.ops import triangulate, unary_union, transform as shapely_transform
from numba import njit

NYC_API = 'https://data.cityofnewyork.us/resource/5zhs-2jue.json'
# fixed study rectangle in Midtown, NYC (W,S,E,N) -- can be changed
CORE_BBOX = (-73.98775, 40.7521, -73.98565, 40.7539)
MATERIAL_ALBEDOS = {'roof': 0.20, 'wall': 0.30, 'ground': 0.12}


def fetch_nyc_buildings(bbox=CORE_BBOX, context_deg=0.001, timeout=75):
    """Fetch geometries by official Socrata v2 API, returning (records, query_metadata).
    Explicitly fail if no rows or truncated pagination. No invented buildings.
    """
    west, south, east, north = bbox
    where = (f'within_box(the_geom,{north+context_deg:.7f},'
             f'{west-context_deg:.7f},{south-context_deg:.7f},'
             f'{east+context_deg:.7f})')
    params = {'$where': where, '$limit': 2000, '$order': 'objectid'}
    r = requests.get(NYC_API, params=params, timeout=timeout)
    r.raise_for_status()
    rows = r.json()
    if isinstance(rows, dict) or not rows:
        raise ValueError(f'No valid NYC footprint records returned: {str(rows)[:500]}')
    if len(rows) >= 2000:
        raise ValueError('NYC query may be truncated; reduce study area or add pagination.')
    records=[]
    for row in rows:
        geom=row.get('the_geom'); height=row.get('height_roof')
        try:
            h=float(height)*0.3048  # official attribute: feet above ground
            polygon=shape(geom)
        except (ValueError, TypeError, AttributeError):
            continue
        if not polygon.is_valid:
            polygon=polygon.buffer(0)
        if not polygon.is_empty and 3 <= h <= 450:
            records.append({'id':str(row.get('doitt_id','')), 'height_m':h,
                            'geometry':polygon,'source':'NYC OTI building footprints',
                            'height_roof_ft':float(height)})
    if not records:
        raise ValueError('NYC query returned rows, but none had usable heights and polygons.')
    return records, {'api':NYC_API,'query':params, 'returned_records':len(rows),
                     'valid_height_records':len(records), 'height_unit':'feet -> meters',
                     'geometry_crs':'EPSG:4326', 'timestamp_utc':datetime.now(ZoneInfo('UTC')).isoformat()}


def synthetic_canyon():
    """Explicitly synthetic geometry ONLY for offline self-tests and preview."""
    return [{'id':'demo_A','height_m':22.,'geometry':box(-73.9875,40.7525,-73.9872,40.7529), 'source':'SYNTHETIC'},
            {'id':'demo_B','height_m':32.,'geometry':box(-73.9869,40.7525,-73.9866,40.7529), 'source':'SYNTHETIC'},
            {'id':'demo_C','height_m':14.,'geometry':box(-73.9875,40.7532,-73.9872,40.75355), 'source':'SYNTHETIC'}]


def _parts(g):
    if isinstance(g,Polygon): return [g]
    if isinstance(g,MultiPolygon): return list(g.geoms)
    if hasattr(g,'geoms'): return [x for x in g.geoms if isinstance(x,Polygon)]
    return []


def _roof_tris(poly):
    """Delaunay triangles clipped to polygon using polygon-covers filter; supports holes."""
    for t in triangulate(poly):
        if poly.covers(t) and t.area>0.03:
            yield t


def build_scene(records, bbox=CORE_BBOX, ground_cell_m=17, simplify_m=0.65,
                max_buildings=65):
    """Build 3D triangle patches; flat ground, flat roofs, vertical walls.
    Coordinates local ENU-like (UTM east/north minus area SW, z above flat terrain).
    """
    transformer=Transformer.from_crs('EPSG:4326','EPSG:32618',always_xy=True)
    ox,oy=transformer.transform(bbox[0],bbox[1])
    def local(g):
        return shapely_transform(lambda x,y,z=None:(np.asarray(transformer.transform(x,y)[0])-ox,
                                              np.asarray(transformer.transform(x,y)[1])-oy),g)
    core=local(box(*bbox))
    converted=[]
    for rec in records:
        pp=local(rec['geometry'])
        if pp.is_empty or not pp.intersects(core.buffer(120)):continue
        for poly in _parts(pp):
            p=poly.simplify(simplify_m,preserve_topology=True)
            if p.is_valid and p.area>=8:
                converted.append((p,float(rec['height_m']),str(rec['id'])))
    if not converted:raise ValueError('No footprints intersect buffered study area.')
    # Retain closest features, not simply first rows from municipal API
    converted.sort(key=lambda x:x[0].distance(core))
    converted=converted[:max_buildings]
    triangles=[]; kinds=[]; ids=[]
    def add(pts,kind,building_id):
        v=np.asarray(pts,dtype=float)
        n=np.cross(v[1]-v[0],v[2]-v[0]); a=np.linalg.norm(n)/2
        if a>1.e-5:
            triangles.append(v);kinds.append(kind);ids.append(building_id)
    for poly,h,bid in converted:
        oriented=orient(poly,sign=1.0)
        for t in _roof_tris(oriented):
            xy=list(t.exterior.coords)[:3]
            add([(xy[0][0],xy[0][1],h),(xy[1][0],xy[1][1],h),(xy[2][0],xy[2][1],h)],'roof',bid)
        for ring in [oriented.exterior]+list(oriented.interiors):
            coords=list(ring.coords)
            for j in range(len(coords)-1):
                x1,y1=coords[j];x2,y2=coords[j+1]
                seg=math.hypot(x2-x1,y2-y1)
                nseg=max(1,int(math.ceil(seg/18)))
                for t in range(nseg):
                    frac0=t/nseg;frac1=(t+1)/nseg
                    x0=x1+(x2-x1)*frac0;y0=y1+(y2-y1)*frac0
                    xx=x1+(x2-x1)*frac1;yy=y1+(y2-y1)*frac1
                    # two outward-facing triangles
                    add([(x0,y0,0),(xx,yy,0),(xx,yy,h)],'wall',bid)
                    add([(x0,y0,0),(xx,yy,h),(x0,y0,h)],'wall',bid)
    # Ground fragments are clipped around footprints; no ground under buildings.
    occupied=unary_union([p for p,_,_ in converted])
    xmin,ymin,xmax,ymax=core.bounds
    nx=max(1,int(np.ceil((xmax-xmin)/ground_cell_m)))
    ny=max(1,int(np.ceil((ymax-ymin)/ground_cell_m)))
    for i in range(nx):
        for j in range(ny):
            left=xmin+(xmax-xmin)*i/nx;right=xmin+(xmax-xmin)*(i+1)/nx
            bottom=ymin+(ymax-ymin)*j/ny;top=ymin+(ymax-ymin)*(j+1)/ny
            free=box(left,bottom,right,top).difference(occupied)
            for piece in _parts(free):
                for tri in _roof_tris(piece):
                    xy=list(tri.exterior.coords)[:3]
                    add([(a,b,0) for a,b in xy], 'ground', 'GROUND')
    v=np.asarray(triangles,dtype=np.float64)
    if not len(v):raise RuntimeError('Scene mesh empty.')
    cross=np.cross(v[:,1,:]-v[:,0,:],v[:,2,:]-v[:,0,:])
    area=np.linalg.norm(cross,axis=1)/2
    normal=cross/(2*area[:,None])
    centers=v.mean(axis=1)
    # Geometry sanity checks
    for i,k in enumerate(kinds):
        if k in ('roof','ground') and normal[i,2]<0.999:
            v[i,[1,2]]=v[i,[2,1]];normal[i]*=-1
    return {'triangles':v,'area':area,'normal':normal,'center':centers,
            'kind':np.array(kinds), 'building_id':np.array(ids),
            'bbox':bbox,'core_polygon':core,
            'building_count':len(set(x[2] for x in converted)),
            'source_is_synthetic':all(x.get('source')=='SYNTHETIC' for x in records),
            'bounds_m':(xmin,ymin,xmax,ymax)}


def solar_position_noaa(dt, lat=40.7530, lon=-73.9868):
    """Approximate NOAA solar position, local tangent axes (east,north,up).

    Inputs: timezone-aware datetime and geographical WGS84 latitude/longitude.
    UTC-based equation of time removes the old hard-coded NYC timezone. This
    remains an approximate point-sun model (not a measured solar ephemeris).
    Returns a direction pointing *from the surface toward the sun*.
    """
    from datetime import timezone
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError('Supply timezone-aware datetime')
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError('Invalid latitude/longitude')
    utc = dt.astimezone(timezone.utc)
    doy = utc.timetuple().tm_yday
    leap = utc.year%4 == 0 and (utc.year%100 != 0 or utc.year%400 == 0)
    hours = utc.hour + utc.minute/60 + (utc.second+utc.microsecond/1e6)/3600
    gamma = 2*np.pi/(366 if leap else 365)*(doy-1+(hours-12)/24)
    eq = 229.18*(0.000075+0.001868*np.cos(gamma)-0.032077*np.sin(gamma)
                 -0.014615*np.cos(2*gamma)-0.040849*np.sin(2*gamma))
    dec = (0.006918-0.399912*np.cos(gamma)+0.070257*np.sin(gamma)
           -0.006758*np.cos(2*gamma)+0.000907*np.sin(2*gamma)
           -0.002697*np.cos(3*gamma)+0.00148*np.sin(3*gamma))
    true_solar_minutes = (hours*60 + eq + 4*lon) % 1440
    ha = np.deg2rad(true_solar_minutes/4-180)
    phi = np.deg2rad(lat)
    east = -np.cos(dec)*np.sin(ha)
    north = np.cos(phi)*np.sin(dec)-np.sin(phi)*np.cos(dec)*np.cos(ha)
    up = np.sin(phi)*np.sin(dec)+np.cos(phi)*np.cos(dec)*np.cos(ha)
    vec = np.array([east,north,up],dtype=float)
    vec /= np.linalg.norm(vec)
    return {'direction':vec,
            'elevation_deg':float(np.rad2deg(np.arcsin(np.clip(vec[2],-1,1)))),
            'azimuth_deg':float(np.rad2deg(np.arctan2(vec[0],vec[1]))%360),
            'approximate_method':'NOAA compact equations, UTC',
            'datetime_utc':utc.isoformat(),
            'latitude_deg':float(lat),'longitude_deg':float(lon)}


@njit(cache=True)
def _hit(origin,direction,triangles,sphere_centers,sphere_radii,ignore=-1):
    best=1e20;idx=-1
    for k in range(triangles.shape[0]):
        if k==ignore:continue
        # Conservative bounding sphere pruning: never drops a true triangle hit.
        wx=sphere_centers[k,0]-origin[0]
        wy=sphere_centers[k,1]-origin[1]
        wz=sphere_centers[k,2]-origin[2]
        projection=wx*direction[0]+wy*direction[1]+wz*direction[2]
        radius=sphere_radii[k]
        if projection < -radius:continue
        dist2=wx*wx+wy*wy+wz*wz-projection*projection
        if dist2>radius*radius:continue
        a=triangles[k,0];b=triangles[k,1];c=triangles[k,2]
        e1=b-a;e2=c-a
        p=np.cross(direction,e2)
        det=e1[0]*p[0]+e1[1]*p[1]+e1[2]*p[2]
        if abs(det)<1e-10:continue
        inv=1.0/det
        t=origin-a
        u=(t[0]*p[0]+t[1]*p[1]+t[2]*p[2])*inv
        if u<0 or u>1:continue
        q=np.cross(t,e1)
        v=(direction[0]*q[0]+direction[1]*q[1]+direction[2]*q[2])*inv
        if v<0 or u+v>1:continue
        dist=(e2[0]*q[0]+e2[1]*q[1]+e2[2]*q[2])*inv
        if 1.e-4<dist<best:best=dist;idx=k
    return idx


@njit(cache=True)
def _cos_sample(n,u,v):
    """Cosine-weighted hemisphere about outward normal (Lambertian distribution)."""
    r=np.sqrt(u); ang=2*np.pi*v
    x=r*np.cos(ang);y=r*np.sin(ang);z=np.sqrt(max(0.,1-u))
    if abs(n[2])<0.9:base=np.array([0.,0.,1.])
    else:base=np.array([1.,0.,0.])
    tangent=np.cross(base,n);tangent=tangent/np.sqrt(np.sum(tangent*tangent))
    bi=np.cross(n,tangent)
    return x*tangent+y*bi+z*n


@njit(cache=True)
def _point_on_tri(tri,r1,r2):
    s=np.sqrt(r1)
    return (1-s)*tri[0]+s*(1-r2)*tri[1]+s*r2*tri[2]


@njit(cache=True)
def _initial_flux(tris,area,normals,centers,sphere_radii,sun,dni,dhi,sky_rays,direct_samples,seed):
    np.random.seed(seed)
    m=len(area)
    direct=np.zeros(m);sky=np.zeros(m);sky_visible=np.zeros(m)
    for i in range(m):
        dot=max(0.,np.sum(normals[i]*sun))
        if dot>0 and sun[2]>0:
            # Sample direct-sun visibility over the entire triangle, instead
            # of assuming the centroid represents an entire roof or facade.
            unobstructed=0
            for r in range(direct_samples):
                xyz=_point_on_tri(tris[i],np.random.random(),np.random.random())
                x=xyz+normals[i]*0.02
                if _hit(x,sun,tris,centers,sphere_radii,i)<0:unobstructed+=1
            direct[i]=dni*dot*area[i]*unobstructed/direct_samples
        unblocked=0
        for r in range(sky_rays):
            xyz=_point_on_tri(tris[i],np.random.random(),np.random.random())
            d=_cos_sample(normals[i],np.random.random(),np.random.random())
            x=xyz+normals[i]*0.02
            if d[2]>0 and _hit(x,d,tris,centers,sphere_radii,i)<0:unblocked+=1
        sky_visible[i]=unblocked/sky_rays
        sky[i]=dhi*area[i]*sky_visible[i]
    return direct,sky,sky_visible


@njit(cache=True)
def _reflect_once(tris,normal,emitted,rays,seed,kind_codes,centers,sphere_radii):
    np.random.seed(seed)
    incoming=np.zeros(len(emitted));escape=0.
    cross=np.zeros((3,3))
    for i in range(len(emitted)):
        power=emitted[i]
        if power<=0:continue
        each=power/rays
        for q in range(rays):
            start=_point_on_tri(tris[i],np.random.random(),np.random.random())+normal[i]*0.02
            d=_cos_sample(normal[i],np.random.random(),np.random.random())
            j=_hit(start,d,tris,centers,sphere_radii,i)
            if j<0:
                escape+=each
            else:
                incoming[j]+=each
                cross[kind_codes[i],kind_codes[j]]+=each
    return incoming,escape,cross


def simulate(scene, dt, *, dni=750.,dhi=120.,albedos=None,
             bounces=3,sky_rays=8,bounce_rays=10,direct_samples=4,seed=42,
             latitude=None,longitude=None):
    """Energy-consistent Monte Carlo approximation for fixed modeled solar inputs.
    Source power = directly + sky-diffuse incident power. Track absorbed, escaped, remainder.
    The sky model is isotropic; surfaces reflect diffusely with constant albedo.
    """
    if albedos is None:albedos=MATERIAL_ALBEDOS
    if not all(0<=v<=1 for v in albedos.values()):raise ValueError('Albedo must be 0..1')
    if dni<0 or dhi<0 or bounces<0 or sky_rays<1 or bounce_rays<1 or direct_samples<1:
        raise ValueError('Invalid simulation settings')
    if latitude is None or longitude is None:
        west,south,east,north=scene['bbox']
        latitude=(south+north)/2 if latitude is None else latitude
        longitude=(west+east)/2 if longitude is None else longitude
    s=solar_position_noaa(dt,lat=latitude,lon=longitude)
    kinds=scene['kind']; kind_codes=np.array([('roof','wall','ground').index(k) for k in kinds],dtype=np.int64)
    alpha=np.asarray([albedos[k] for k in kinds],dtype=float)
    sphere_radii=np.max(np.linalg.norm(scene['triangles']-scene['center'][:,None,:],axis=2),axis=1)
    direct,sky,skyvis=_initial_flux(scene['triangles'],scene['area'],scene['normal'],
                                  scene['center'],sphere_radii,s['direction'],dni,dhi,sky_rays,direct_samples,seed)
    initial=direct+sky
    incoming=initial.copy()
    absorbed=np.zeros_like(initial)
    total_escape=0.;transfer=np.zeros((3,3))
    per_bounce=[]
    for b in range(bounces+1):
        absorbed+=(1-alpha)*incoming
        outgoing=alpha*incoming
        per_bounce.append({'bounce':b,'incident_W':float(incoming.sum()),
                           'absorbed_W':float(((1-alpha)*incoming).sum()),
                           'reflected_W':float(outgoing.sum())})
        if b==bounces:
            unresolved=float(outgoing.sum());break
        incoming,escaped,cross=_reflect_once(scene['triangles'],scene['normal'],outgoing,
                                       bounce_rays,seed+31*(b+1),kind_codes,scene['center'],sphere_radii)
        total_escape+=escaped;transfer+=cross
    power0=float(initial.sum())
    absorbed_w=float(absorbed.sum())
    delta=power0-(absorbed_w+total_escape+unresolved)
    return {'absorbed_W':absorbed,'absorbed_Wm2':absorbed/scene['area'],
            'incident_direct_W':direct,'incident_sky_W':sky,
            'sky_visible_fraction':skyvis,
            'initial_W':initial,'albedo':alpha,
            'reflection_transfer_W':transfer,'bounces':per_bounce,
            'budget':{'initial_W':power0,'absorbed_W':absorbed_w,
                       'escaped_reflection_W':float(total_escape),
                       'unresolved_reflection_W':unresolved,'residual_W':float(delta),
                       'relative_error':float(abs(delta)/max(1.,power0))},
            'solar':{k:v for k,v in s.items() if k!='direction'},
            'sun_vector':s['direction'],
            'assumptions':{'dni_Wm2':dni,'dhi_Wm2':dhi,'sky_model':'isotropic',
                           'reflection':'Lambertian diffuse', 'direct_samples_per_triangle':direct_samples, 'temperature_calculated':False}}


def simulation_frame(scene,result,scenario_id):
    from pandas import DataFrame
    s=result['sun_vector']
    rows=DataFrame({
      'scenario_id':scenario_id,'face_index':np.arange(len(scene['area'])),
      'kind':scene['kind'],'building_id':scene['building_id'],
      'x':scene['center'][:,0],'y':scene['center'][:,1],'z':scene['center'][:,2],
      'nx':scene['normal'][:,0],'ny':scene['normal'][:,1],'nz':scene['normal'][:,2],
      'area_m2':scene['area'],'albedo':result['albedo'],
      'solar_east':s[0],'solar_north':s[1],'solar_up':s[2],
      'dni':result['assumptions']['dni_Wm2'],'dhi':result['assumptions']['dhi_Wm2'],
      'direct_Wm2':result['incident_direct_W']/scene['area'],
      'sky_Wm2':result['incident_sky_W']/scene['area'],
      'sky_view':result['sky_visible_fraction'],
      'no_reflection_Wm2':(1-result['albedo'])*result['initial_W']/scene['area'],
      'multibounce_Wm2':result['absorbed_Wm2']
    })
    return rows


def scene_json_metadata(scene):
    return {'building_count':scene['building_count'], 'num_surface_triangles':len(scene['area']),
            'surface_area_m2':{k:float(scene['area'][scene['kind']==k].sum()) for k in ('roof','wall','ground')},
            'source_is_synthetic':bool(scene['source_is_synthetic']),
            'bbox_lon_lat':list(scene['bbox'])}


def render_html(scene,result,path,title='UrbanRay AI — absorbed solar power'):
    import plotly.graph_objects as go
    V=scene['triangles'].reshape(-1,3)
    n=len(scene['area']);faces=np.arange(3*n).reshape(-1,3)
    intensity=np.minimum(result['absorbed_Wm2'],1500)
    mesh=go.Mesh3d(x=V[:,0],y=V[:,1],z=V[:,2],
                   i=faces[:,0],j=faces[:,1],k=faces[:,2],
                   intensity=intensity,intensitymode='cell',colorscale='Inferno',
                   cmin=0,cmax=max(100,float(np.percentile(intensity,98))),
                   colorbar={'title':'W/m² absorbed'},flatshading=True,
                   lighting={'ambient':0.7,'diffuse':0.6},name='Surface flux')
    fig=go.Figure(data=[mesh]);fig.update_layout(title=title,
        scene=dict(aspectmode='data',xaxis_title='East (m)',yaxis_title='North (m)',zaxis_title='Height (m)'),
        margin=dict(l=0,r=0,b=0,t=55),height=700)
    fig.write_html(str(path),include_plotlyjs=True)
    return fig


def make_ml_surrogate(df,seed=123):
    """AI physics-surrogate experiment with no random face-level leakage.

    Split by whole dates: first two dates train, third validate,
    fourth untouched test. Select model on validation only.
    Ground truth is modeled 3-bounce flux, not observed surface temperature.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    df=df.copy()
    df['date_key']=df['scenario_id'].str.slice(0,8)
    dates=sorted(df['date_key'].unique())
    if len(dates)!=4:
        raise ValueError('Need exactly four study dates for chronological train/val/test split')
    train=df[df.date_key.isin(dates[:2])].copy()
    val=df[df.date_key==dates[2]].copy()
    test=df[df.date_key==dates[3]].copy()
    features=['kind','x','y','z','nx','ny','nz','area_m2','albedo',
      'solar_east','solar_north','solar_up','dni','dhi','direct_Wm2','sky_Wm2','sky_view']
    xall=pd.get_dummies(df[features],columns=['kind'],dtype=float)
    xtr=xall.loc[train.index];xva=xall.loc[val.index];xts=xall.loc[test.index]
    ycorr_train=train['multibounce_Wm2']-train['no_reflection_Wm2']
    candidates={
      'HistogramGradientBoosting': HistGradientBoostingRegressor(
         max_iter=175,max_leaf_nodes=15,l2_regularization=10,random_state=seed),
      'ExtraTrees': ExtraTreesRegressor(n_estimators=160,min_samples_leaf=2,
                             max_features=1.0,n_jobs=-1,random_state=seed)}
    scores={}
    for name,model in candidates.items():
        model.fit(xtr,ycorr_train)
        residual=np.maximum(0,model.predict(xva))
        prediction=np.maximum(0,val['no_reflection_Wm2'].to_numpy()+residual)
        scores[name]=float(mean_absolute_error(val['multibounce_Wm2'],prediction))
    chosen=min(scores,key=scores.get)
    chosen_model=candidates[chosen]
    combined=pd.concat([train,val],axis=0)
    ycorr=combined['multibounce_Wm2']-combined['no_reflection_Wm2']
    chosen_model.fit(xall.loc[combined.index],ycorr)
    residual=np.maximum(0,chosen_model.predict(xts))
    pred=np.maximum(0,test['no_reflection_Wm2'].to_numpy()+residual)
    truth=test['multibounce_Wm2'].to_numpy()
    base=test['no_reflection_Wm2'].to_numpy()
    metrics={
      'training_dates':dates[:2],'validation_date':dates[2],
      'untouched_test_date':dates[3],
      'candidate_validation_mae_Wm2':scores,'selected_model':chosen,
      'test_rows':int(len(test)),
      'baseline_no_reflections_mae_Wm2':float(mean_absolute_error(truth,base)),
      'ML_residual_surrogate_mae_Wm2':float(mean_absolute_error(truth,pred)),
      'ML_R2_vs_3bounce_simulator':float(r2_score(truth,pred)),
      'model_target':'3-bounce radiation-simulator output; NOT observed temperature',
      'important_limit':'Same geometry used across dates; does not demonstrate generalization to a different city or block.'}
    predictions=test[['date_key','scenario_id','face_index','kind','no_reflection_Wm2','multibounce_Wm2']].copy()
    predictions['ai_predicted_Wm2']=pred
    return chosen_model,features,xall.columns.tolist(),metrics,predictions


def render_scenario_comparison(scene,standard,reflective,path):
    """Self-contained Plotly dropdown for 3D flux, roof albedo scenarios and difference."""
    import plotly.graph_objects as go
    V=scene['triangles'].reshape(-1,3)
    f=np.arange(len(V)).reshape(-1,3)
    arrays=[standard['absorbed_Wm2'],reflective['absorbed_Wm2'],
            reflective['absorbed_Wm2']-standard['absorbed_Wm2']]
    names=['Baseline roof alpha=0.18','Reflective roof alpha=0.58','Change: reflective minus baseline']
    fig=go.Figure()
    for i,z in enumerate(arrays):
        cmax=max(25,float(np.percentile(np.abs(z),98))) if i==2 else max(100,float(np.percentile(z,98)))
        fig.add_trace(go.Mesh3d(x=V[:,0],y=V[:,1],z=V[:,2],
           i=f[:,0],j=f[:,1],k=f[:,2], intensity=z,intensitymode='cell',
           colorscale='RdBu_r' if i==2 else 'Inferno',
           cmin=-cmax if i==2 else 0,cmax=cmax,
           flatshading=True,visible=(i==0),name=names[i],
           colorbar={'title':'Change W/m²' if i==2 else 'Absorbed W/m²'}))
    buttons=[]
    for i,name in enumerate(names):
        buttons.append(dict(label=name,method='update',args=[{'visible':[j==i for j in range(3)]},
                                                       {'title':'UrbanRay AI | '+name}]))
    fig.update_layout(title='UrbanRay AI | '+names[0],
       updatemenus=[dict(type='dropdown',direction='down',buttons=buttons,x=0.02,y=1.07)],
       scene=dict(aspectmode='data',xaxis_title='East (m)',yaxis_title='North (m)',zaxis_title='Height (m)'),
       height=720,margin=dict(l=0,r=0,b=0,t=75))
    fig.write_html(str(path),include_plotlyjs=True)
    return fig


def render_reflection_sankey(result,path):
    """Power transfers W across roof / wall / ground, all modeled reflection bounces."""
    import plotly.graph_objects as go
    types=['roof','wall','ground'];sources=[];targets=[];vals=[]
    flows=result['reflection_transfer_W']
    for i in range(3):
        for j in range(3):
            if flows[i,j]>0:
                sources.append(i);targets.append(3+j);vals.append(float(flows[i,j]))
    fig=go.Figure(go.Sankey(node=dict(label=['Reflected by '+x for x in types]+
                                 ['Received by '+x for x in types],pad=25,thickness=15),
                           link=dict(source=sources,target=targets,value=vals)))
    fig.update_layout(title='Inter-surface radiation transfers (modeled reflection bounces)',height=480)
    fig.write_html(str(path),include_plotlyjs=True)
    return fig

def fetch_openmeteo_radiation(dt, lat=40.7530, lon=-73.9868, timeout=50):
    """Fetch historical *modeled* hour-specific DNI/DHI (instantaneous, W/m²) from Open-Meteo.
    IMPORTANT: Open-Meteo archive values are gridded model/reanalysis estimates, not local radiometer readings.
    Solar position uses the requested local hour. If record is absent, fail rather than fabricate.
    """
    if dt.tzinfo is None:
        raise ValueError('Timezone-aware datetime required')
    local=dt.astimezone(ZoneInfo('America/New_York'))
    day=local.strftime('%Y-%m-%d')
    params={'latitude':lat,'longitude':lon,'start_date':day,'end_date':day,
            'hourly':'direct_normal_irradiance_instant,diffuse_radiation_instant',
            'timezone':'America/New_York'}
    url='https://archive-api.open-meteo.com/v1/archive'
    response=requests.get(url,params=params,timeout=timeout)
    response.raise_for_status()
    raw=response.json()
    h=raw.get('hourly',{})
    times=h.get('time',[])
    key=local.strftime('%Y-%m-%dT%H:00')
    if key not in times:raise ValueError(f'Requested time {key} not available in Open-Meteo archive')
    idx=times.index(key)
    dni=h.get('direct_normal_irradiance_instant',[None]*len(times))[idx]
    dhi=h.get('diffuse_radiation_instant',[None]*len(times))[idx]
    if dni is None or dhi is None or not np.isfinite(dni) or not np.isfinite(dhi):
        raise ValueError(f'Missing DNI/DHI in Open-Meteo response for {key}')
    if dni<0 or dhi<0:raise ValueError('Negative irradiance from weather data')
    return {'dni':float(dni),'dhi':float(dhi),'source':url,
            'source_type':'modeled gridded historical irradiance',
            'timestamp_local':key,'request_parameters':params}
