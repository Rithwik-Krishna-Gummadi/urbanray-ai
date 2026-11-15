"""UrbanRay Physics+ — independently audited *modeled* radiative / thermal transport.

Inputs:
  Real NYC footprint locations/heights, simplified polygon-prism buildings.
  Weather: gridded *model* outputs, not onsite observations.
  Material properties: explicit scenarios, NOT observations of real buildings.

Units: triangle geometry m; surface temperatures kelvin; shortwave W/m2;
longwave surface exchange W/m2. A sparse, reciprocal diffuse view-factor
operator is estimated by geometric Monte Carlo, but this is a model of ideal
Lambertian surfaces and not a validated urban microclimate model.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, diags
from numba import njit
from urbanray import _hit, _point_on_tri, _cos_sample
from integrated88 import SIGMA, convection_h, sky_emissivity

@dataclass(frozen=True)
class ThermalMaterial:
    alpha: float
    emissivity: float
    k: float                         # W/m/K, assumed effective
    volume_heat_capacity: float      # J/m3/K, assumed effective
    skin_depth_m: float              # m, assumed
    substrate_depth_m: float         # m, assumed
    backside_h: float                # W/m2/K, assumed

ASSUMED_MATERIALS = {
    'roof': ThermalMaterial(.18, .91, .85, 1.55e6, .065, .25, .9),
    'wall': ThermalMaterial(.30, .92, 1.25, 1.80e6, .09, .30, 1.0),
    'ground': ThermalMaterial(.12, .95, .76, 1.65e6, .11, .45, 0.5)
}

@njit(cache=True)
def _sample_exchanges(tris, normals, centers, sphere_radii, rays, seed):
    """Unbiased cosine-weighted emission sampling per patch; return individual hits.

    This form samples view factors without building the dense n*n matrix.
    """
    np.random.seed(seed)
    n=len(tris)
    origins=np.empty(n*rays,dtype=np.int32)
    targets=np.empty(n*rays,dtype=np.int32)
    for i in range(n):
        for j in range(rays):
            k=i*rays+j
            pt=_point_on_tri(tris[i],np.random.random(),np.random.random())
            direction=_cos_sample(normals[i],np.random.random(),np.random.random())
            hit=_hit(pt+normals[i]*.02,direction,tris,centers,sphere_radii,i)
            origins[k]=i
            targets[k]=hit
    return origins,targets


def build_reciprocal_operator(scene,rays=12,seed=27):
    """Area-reciprocal, row-substochastic diffuse transport P[i,j].

    P[i,j] is fraction of radiant power leaving patch i hitting patch j;
    P[i,sky] = 1 - sum(P[i,:]). reciprocity: A_i P_ij = A_j P_ji.

    Reciprocity is imposed after Monte Carlo, creating an *approximate*
    view-factor operator and not exact geometric transport. This symmetrization
    can bias individual transfers, particularly for small adjacent patches.
    """
    if not (1 <= rays <= 10000): raise ValueError('rays must be 1..10000')
    area=np.asarray(scene['area'],dtype=float)
    if not np.all(area>0): raise ValueError('Area must be positive')
    tris=np.ascontiguousarray(scene['triangles'],dtype=np.float64)
    centers=np.ascontiguousarray(scene['center'],dtype=np.float64)
    normals=np.ascontiguousarray(scene['normal'],dtype=np.float64)
    radii=np.max(np.linalg.norm(tris-centers[:,None,:],axis=2),axis=1)
    origins,targets=_sample_exchanges(tris,normals,centers,radii,rays,seed)
    good=targets>=0
    n=len(area)
    counts=csr_matrix((np.ones(good.sum(),dtype=float),
                       (origins[good],targets[good])), shape=(n,n))
    P0=counts*(1./rays)
    Q=(diags(area)@P0 + (diags(area)@P0).T)*.5
    # Symmetric conservation scaling: each patch must emit <= area-weighted 1.
    row_sum=np.asarray(Q.sum(axis=1)).ravel()
    s=np.minimum(1.,np.divide(area,row_sum,out=np.ones_like(area),where=row_sum>0))
    Q=(diags(s)@Q@diags(s)).tocsr()
    P=(diags(1./area)@Q).tocsr()
    P.eliminate_zeros()
    sky=np.maximum(0.,1.-np.asarray(P.sum(axis=1)).ravel())
    reciprocity=Q-Q.T
    reciprocity.max() if reciprocity.nnz else None
    max_rec=float(np.max(np.abs(reciprocity.data))) if reciprocity.nnz else 0.
    if np.max(np.asarray(P.sum(axis=1)).ravel())>1+1e-9:
        raise AssertionError('Radiative power creation in view factor operator')
    return P,sky,{'rays_per_surface':rays,'seed':seed,'num_patches':n,
                 'nonzero_links':int(P.nnz),'max_reciprocity_power_area_residual_m2':max_rec,
                 'sky_fraction_area_weighted':float(np.average(sky,weights=area)),
                 'source':'geometric Monte Carlo + area reciprocity symmetrization'}


def thermal_properties(kinds,materials=None):
    mat=ASSUMED_MATERIALS if materials is None else materials
    result={}
    for name,attr in [('alpha','alpha'),('emissivity','emissivity'),('Cs','skin_depth_m'),
                      ('Cd','substrate_depth_m'),('Gsd','k'),('Gb','backside_h')]:
        if name in ('Cs','Cd'):
            result[name]=np.array([getattr(mat[k],attr)*mat[k].volume_heat_capacity for k in kinds],dtype=float)
        elif name=='Gsd':
            result[name]=np.array([mat[k].k/max(1e-6,.5*(mat[k].skin_depth_m+mat[k].substrate_depth_m)) for k in kinds],dtype=float)
        else:
            result[name]=np.array([getattr(mat[k],attr) for k in kinds],dtype=float)
    if np.any(result['Cs']<=0) or np.any(result['Cd']<=0):
        raise ValueError('heat capacity must be positive')
    if np.any((result['alpha']<0)|(result['alpha']>1)) or np.any((result['emissivity']<0)|(result['emissivity']>1)):
        raise ValueError('invalid optical properties')
    return result


def longwave_exchange(temperature_K, P, sky_fraction, area, emissivity, sky_flux_Wm2,
                      max_iter=80, tolerance=1e-7):
    """Diffuse-gray radiosity. Returns net longwave on each surface (W/m²).

    Each surface: J = epsilon*sigma*T^4 + (1-epsilon)*H.
    H = P.T@(A*J)/A + sky_fraction * L_sky.
    The gray model treats diffuse IR independently from the shortwave albedo.
    """
    T=np.asarray(temperature_K,dtype=float)
    eps=np.asarray(emissivity,dtype=float)
    area=np.asarray(area,dtype=float)
    sky=np.asarray(sky_fraction,dtype=float)
    if not np.all(np.isfinite(T)) or np.any(T<=0):raise ValueError('invalid temperature')
    if not np.all((eps>=0)&(eps<=1)):raise ValueError('invalid emissivity')
    emitted=eps*SIGMA*T**4
    J=SIGMA*T**4
    for step in range(max_iter):
        H=np.asarray(P.T@(J*area)).ravel()/area + sky*sky_flux_Wm2
        next_J=emitted+(1-eps)*H
        if np.max(np.abs(next_J-J))<=tolerance:
            J=next_J
            break
        J=next_J
    else:
        raise RuntimeError('Longwave radiosity did not converge')
    H=np.asarray(P.T@(J*area)).ravel()/area + sky*sky_flux_Wm2
    net=eps*(H-SIGMA*T**4)
    if not np.all(np.isfinite(net)):raise RuntimeError('Nonfinite longwave flux')
    return net,{'iterations':step+1,'longwave_net_W':float(np.dot(net,area)),
                'sky_flux_Wm2':float(sky_flux_Wm2)}


def step_two_layer(Ts_prev,Td_prev,solar_Wm2,air_K,wind_ms,rh_percent,cloud_fraction,
                   P,sky,scene,materials=None,dt=3600,deep_reference_K=None,max_iterations=35):
    """Semi-implicit two-node surface/substrate heat balance, temperature in K.

    Coupled surface radiosity is solved self-consistently in a damped iteration.
    Deep boundary temperature: supplied stable climatological or assumed fixed
    value; setting to ambient each hour makes the heat capacity less realistic.
    """
    kinds=scene['kind'];area=scene['area']
    mat=thermal_properties(kinds,materials)
    Ts0=np.asarray(Ts_prev,dtype=float); Td0=np.asarray(Td_prev,dtype=float)
    SW=np.asarray(solar_Wm2,dtype=float)
    air=float(air_K)
    if deep_reference_K is None: deep_reference_K=air
    Tref=float(deep_reference_K)
    C,CD,G,Gb,eps=mat['Cs'],mat['Cd'],mat['Gsd'],mat['Gb'],mat['emissivity']
    if not (np.all(np.isfinite(SW)) and np.all(SW>=-1e-6)):
        raise ValueError('negative or nonfinite shortwave')
    # sky emissivity scheme is approximate; clamp to physically admissible values
    Lsky,sky_eps=sky_emissivity(air,rh_percent,cloud_fraction)
    if not (0<Lsky<1000):raise ValueError('Unphysical sky longwave flux')
    Ts=Ts0.copy();Td=Td0.copy()
    d=1.0/max(1.,dt)
    for iteration in range(max_iterations):
        lw,_=longwave_exchange(Ts,P,sky,area,eps,Lsky)
        h=convection_h(Ts,air,wind_ms,scene['center'][:,2],kinds)
        # Explicit net longwave from latest estimate + implicit time, convection,
        # conduction. Damping stabilizes nonlinear feedback.
        numerator=C*d*Ts0 + SW + lw + h*air + G*Td
        denominator=C*d + h + G
        updated_s=numerator/denominator
        updated_d=(CD*d*Td0+G*updated_s+Gb*Tref)/(CD*d+G+Gb)
        updated_s=.55*updated_s+.45*Ts
        updated_d=.55*updated_d+.45*Td
        if np.max(np.abs(updated_s-Ts))<3e-4 and np.max(np.abs(updated_d-Td))<3e-4:
            Ts,Td=updated_s,updated_d
            break
        Ts,Td=updated_s,updated_d
    else:
        raise RuntimeError('Thermal iteration did not converge')
    if np.min(Ts)<180 or np.max(Ts)>410 or not np.all(np.isfinite(Ts)):
        raise RuntimeError('Nonphysical modeled surface temperature')
    # Residual tests evaluated at final iterate
    lw,lw_meta=longwave_exchange(Ts,P,sky,area,eps,Lsky)
    h=convection_h(Ts,air,wind_ms,scene['center'][:,2],kinds)
    surface_res=C*(Ts-Ts0)/dt-SW-lw-h*(air-Ts)-G*(Td-Ts)
    deep_res=CD*(Td-Td0)/dt-G*(Ts-Td)+Gb*(Td-Tref)
    return Ts,Td,{'max_surface_residual_Wm2':float(np.max(np.abs(surface_res))),
                 'max_deep_residual_Wm2':float(np.max(np.abs(deep_res))),
                 'LW_net_MW':lw_meta['longwave_net_W']/1e6,
                 'LW_sky_flux_Wm2':Lsky, 'iterations':iteration+1,
                 'mean_roof_C':float(np.average(Ts[kinds=='roof']-273.15, weights=area[kinds=='roof']))}


def materials_with_roof_alpha(alpha,base=None):
    base=dict(ASSUMED_MATERIALS if base is None else base)
    d=asdict(base['roof']);d['alpha']=float(alpha)
    base['roof']=ThermalMaterial(**d)
    return base


def sensitivity_materials(factor=1.0, emissivity_change=0.0, conductivity_factor=1.0, base=None):
    base=ASSUMED_MATERIALS if base is None else base
    return {k:ThermalMaterial(v.alpha,float(np.clip(v.emissivity+emissivity_change,.7,.99)),
            v.k*conductivity_factor,v.volume_heat_capacity*factor,v.skin_depth_m,v.substrate_depth_m,
            v.backside_h) for k,v in base.items()}

def shortwave_from_operator(source_W, P, sky, area, alpha, bounces=3):
    """Energy-tracking shortwave exchange using same geometric operator each hour.

    Returns W per surface, and per-surface W/m². The unpropagated part of the
    last bounce is recorded as unresolved, not spuriously counted absorbed.
    """
    src=np.asarray(source_W,dtype=float)
    area=np.asarray(area,dtype=float)
    alpha=np.asarray(alpha,dtype=float)
    if np.any(src<0) or np.any((alpha<0)|(alpha>1)) or bounces<0:
        raise ValueError('invalid shortwave power/albedo')
    incoming=src.copy(); absorbed=np.zeros_like(src);escaped=0.
    for b in range(bounces+1):
        absorbed+=(1-alpha)*incoming
        leftover=alpha*incoming
        if b==bounces:
            unresolved=float(leftover.sum()); break
        escaped+=float(np.dot(leftover,sky))
        incoming=np.asarray(P.T@leftover).ravel()
    closure=float(src.sum()-absorbed.sum()-escaped-unresolved)
    if abs(closure)>1e-7*max(1,float(src.sum())):
        raise AssertionError('Shortwave energy budget failed')
    return absorbed/area, {'initial_W':float(src.sum()),
        'absorbed_W':float(absorbed.sum()),'escaped_W':escaped,
        'unresolved_W':unresolved,'budget_residual_W':closure,
        'relative_residual':abs(closure)/max(1,float(src.sum()))}


def solar_source_from_weather(scene,timestamp,dni,dhi,sky,shadow_samples=4,seed=17,
                              latitude=None,longitude=None):
    """New direct illumination + view-factor-based diffuse sky for this hour.

    The geometry is still flat roofs / prism walls; cloud effects enter from
    sourced DNI/DHI, not any invented detailed spatial cloud map.
    """
    from urbanray import _initial_flux,solar_position_noaa
    bbox=scene['bbox']
    lat=(bbox[1]+bbox[3])*.5 if latitude is None else latitude
    lon=(bbox[0]+bbox[2])*.5 if longitude is None else longitude
    sun=solar_position_noaa(timestamp,lat=lat,lon=lon)['direction']
    triangles=scene['triangles'];area=scene['area'];centers=scene['center']
    radii=np.max(np.linalg.norm(triangles-centers[:,None,:],axis=2),axis=1)
    direct,_,_= _initial_flux(np.ascontiguousarray(triangles),
                             np.ascontiguousarray(area),
                             np.ascontiguousarray(scene['normal']),
                             np.ascontiguousarray(centers),
                             np.ascontiguousarray(radii),
                             np.ascontiguousarray(sun),
                             float(dni),0.,1,int(shadow_samples),int(seed))
    return direct + float(dhi)*area*sky, sun
