import sys
from pathlib import Path
import numpy as np
import pytest
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))
from integrated88 import load_verified_geometry, thermal_coefficients, thermal_step, sky_emissivity

def test_source_88_real_records():
    buildings,digest=load_verified_geometry(R/'data'/'nyc_88_source.geojson')
    assert len(buildings)==88 and len(digest)==64
    assert all(b['source']=='NYC OTI footprint snapshot' for b in buildings)

def test_coefficients_positive():
    c,g,e=thermal_coefficients(['roof','wall','ground'])
    assert np.all(c>0) and np.all(g>0) and np.all((e>0)&(e<=1))

def test_no_sun_and_equilibrium_stable():
    kinds=np.array(['roof','wall','ground'])
    a=thermal_step(np.full(3,300.),np.zeros(3),300.,3.,60.,.2,np.ones(3),kinds,np.array([30,5,0]))
    assert np.max(abs(a-300.))<3

def test_more_sun_increases_roof_temperature():
    kinds=np.array(['roof'])
    lo=thermal_step(np.array([300.]),np.array([0.]),300.,3.,60.,.25,np.ones(1),kinds,np.array([25.]))
    hi=thermal_step(np.array([300.]),np.array([600.]),300.,3.,60.,.25,np.ones(1),kinds,np.array([25.]))
    assert hi[0]>lo[0]

def test_cloud_longwave_ordering():
    clear=sky_emissivity(300,60,0)[0]
    cloud=sky_emissivity(300,60,1)[0]
    assert cloud>=clear>0
