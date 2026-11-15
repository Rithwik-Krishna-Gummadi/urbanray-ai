import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from urbanray import solar_position_noaa, synthetic_canyon, build_scene, simulate


def test_solar_position_timezone_invariance():
    t=datetime(2025,6,21,17,tzinfo=timezone.utc)
    a=solar_position_noaa(t,40.753,-73.9868)
    b=solar_position_noaa(t.astimezone(timezone(timedelta(hours=-4))),40.753,-73.9868)
    assert np.allclose(a['direction'],b['direction'],atol=1e-12)
    assert np.isclose(np.linalg.norm(a['direction']),1)

def test_no_nyc_timezone_coupling():
    d=datetime(2025,3,20,12,tzinfo=timezone.utc)
    a=solar_position_noaa(d,0,0)
    # Close to equinox noon at Greenwich, sun should be near zenith.
    assert a['elevation_deg']>85
    assert solar_position_noaa(d,0,120)['elevation_deg']<0

def test_nyc_solstice_angles():
    a=solar_position_noaa(datetime(2025,6,21,17,tzinfo=timezone.utc),40.753,-73.9868)
    assert 70<a['elevation_deg']<75
    assert 150<a['azimuth_deg']<220

def test_bad_geography_and_naive_datetime():
    with pytest.raises(ValueError):solar_position_noaa(datetime(2025,6,21,12),40,-73)
    with pytest.raises(ValueError):solar_position_noaa(datetime(2025,6,21,12,tzinfo=timezone.utc),120,-73)

def test_scene_center_default_equals_explicit_and_ray_budget():
    scene=build_scene(synthetic_canyon(),ground_cell_m=90,max_buildings=3)
    dt=datetime(2025,6,21,17,tzinfo=timezone.utc)
    a=simulate(scene,dt,dni=800,dhi=120,sky_rays=3,bounce_rays=3,direct_samples=2,seed=17)
    bb=scene['bbox']
    b=simulate(scene,dt,dni=800,dhi=120,sky_rays=3,bounce_rays=3,direct_samples=2,seed=17,
               latitude=(bb[1]+bb[3])/2,longitude=(bb[0]+bb[2])/2)
    assert np.allclose(a['absorbed_W'],b['absorbed_W'])
    assert a['budget']['relative_error']<1e-12
    assert len(a['absorbed_W'])==len(scene['area'])

def test_custom_location_changes_sun_direction():
    scene=build_scene(synthetic_canyon(),ground_cell_m=90,max_buildings=3)
    t=datetime(2025,6,21,17,tzinfo=timezone.utc)
    a=simulate(scene,t,dni=750,dhi=100,bounces=0,sky_rays=2,direct_samples=2,bounce_rays=2,seed=3)
    b=simulate(scene,t,dni=750,dhi=100,bounces=0,sky_rays=2,direct_samples=2,bounce_rays=2,seed=3,latitude=0,longitude=0)
    assert not np.allclose(a['sun_vector'],b['sun_vector'])
    assert b['budget']['relative_error']<1e-12
