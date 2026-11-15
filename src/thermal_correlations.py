"""Thermal correlations used by the current 88-building model.

Preserved unchanged from the original shared module; legacy synthetic city removed.
"""
import numpy as np

STEFAN = 5.670374419e-8

def air_properties(T_k):
    t=np.asarray(T_k,dtype=float)
    mu=1.716e-5*(t/273.15)**1.5*(273.15+110.4)/(t+110.4)
    rho=101325./(287.058*t)
    k=.0241*(t/273.15)**.9
    pr=.71+np.zeros_like(t)
    return rho,mu,k,pr

def convection_coefficient(surface_T, air_T, wind_speed, length, vertical=True):
    """Forced flat-plate Nu plus Churchill-Chu free Nu with cubic combination.
    Approximation: exposure and canyon turbulence not explicitly resolved.
    """
    tf=np.clip((np.asarray(surface_T)+np.asarray(air_T))/2,240,360)
    rho,mu,k,pr=air_properties(tf)
    U=np.asarray(wind_speed)
    L=np.maximum(.1,np.asarray(length))
    re=np.maximum(0.,rho*U*L/mu)
    nu_lam=.664*np.sqrt(re)*pr**(1/3)
    nu_turb=np.maximum(0,(.037*np.maximum(re,1)**.8-871))*pr**(1/3)
    nu_forced=np.where(re<=5e5,nu_lam,nu_turb)
    beta=1/tf
    nu_kin=mu/rho;alpha=k/(rho*1005.)
    ra=9.81*beta*np.abs(np.asarray(surface_T)-np.asarray(air_T))*L**3/(nu_kin*alpha)
    nu_free=(.825+(.387*np.maximum(ra,0)**(1/6))/((1+(.492/pr)**(9/16))**(8/27)))**2
    # horizontal surface free-convection: simpler orientation-sensitive correlation
    if isinstance(vertical,np.ndarray):
        horizontal=np.where(np.asarray(surface_T)>=np.asarray(air_T),.54,.27)*np.maximum(ra,0)**.25
        nu_free=np.where(vertical,nu_free,horizontal)
    elif not vertical:
        nu_free=np.where(np.asarray(surface_T)>=np.asarray(air_T),.54,.27)*np.maximum(ra,0)**.25
    hf=nu_forced*k/L;hn=nu_free*k/L
    return (hf**3+hn**3)**(1/3)

def brunt_cloud_longwave(T_air_k, relative_humidity, cloud_fraction):
    """Empirical Brunt clear-sky emissivity + explicit approximate cloud enhancement."""
    tc=float(T_air_k)-273.15
    es_hpa=6.112*np.exp(17.67*tc/(tc+243.5))
    e_hpa=np.clip(float(relative_humidity),0,1)*es_hpa
    eps_clear=np.clip(.51+.066*np.sqrt(e_hpa),.45,.97)
    eps_sky=min(1.,eps_clear+(.98-eps_clear)*float(np.clip(cloud_fraction,0,1))**2)
    return float(eps_sky*STEFAN*T_air_k**4), float(eps_sky)
