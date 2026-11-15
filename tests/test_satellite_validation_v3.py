"""No satellite readings are fabricated here: synthetic unit fixtures only."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer
from shapely.geometry import Polygon

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validate_landsat_pixels import (pixel_polygon,quality_reason,
                                    radiance_proxy_C,pixel_estimate,
                                    reconstructed_scene,interpolate_faces)
from plan_landsat_runs import plan
from shapely.strtree import STRtree


def good_sample():
    lon,lat=-73.9867,40.753
    return dict(lst_c=30.0,qa_pixel=64,qa_radsat=0,st_uncertainty_k=1.2,
       cloud_distance_km=4.0,center_lon=lon,center_lat=lat,
       projection_crs='EPSG:32618',projection_affine='30;0;300000;0;-30;4500000')

def test_qa_masks_and_uncertainty():
    r=good_sample()
    assert quality_reason(r) is None
    for bit in (0,1,2,3,4,5,7):
        assert quality_reason({**r,'qa_pixel':64|(1<<bit)})=='qa_pixel_bad'
    assert quality_reason({**r,'st_uncertainty_k':4})=='st_uncertainty_too_high'
    assert quality_reason({**r,'cloud_distance_km':.1})=='near_cloud'
    assert quality_reason({**r,'qa_radsat':1})=='saturation_or_occlusion'

def test_pixel_area_from_native_affine():
    p=pixel_polygon(good_sample())
    assert p.is_valid and abs(p.area-900)<0.1

def test_radiance_equivalent_equals_uniform_temperature():
    assert abs(radiance_proxy_C([35,35],[3,8],[.9,.95])-35)<1.e-8
    high=radiance_proxy_C([20,40],[1,1],[.9,.9])
    assert high>30 # T^4 weighting differs from arithmetic mean

def test_pixel_polygon_overlap_known_synthetic():
    a=Polygon([(0,0),(30,0),(30,30),(0,30)])
    face1=Polygon([(0,0),(15,0),(15,30),(0,30)])
    face2=Polygon([(15,0),(30,0),(30,30),(15,30)])
    fake_scene={'kind':np.array(['roof','ground'])}
    pred,reason=pixel_estimate(a,a,STRtree([face1,face2]),np.array([0,1]),
        fake_scene,np.array([20.,40.]),np.array([.9,.9]))
    assert reason is None
    assert abs(pred['modeled_area_mean_C']-30)<1e-8
    assert abs(pred['modeled_roof_fraction']-.5)<1e-8
    bad=Polygon([(20,0),(50,0),(50,30),(20,30)])
    pred,reason=pixel_estimate(bad,a,STRtree([face1,face2]),np.array([0,1]),
        fake_scene,np.array([20.,40.]),np.array([.9,.9]))
    assert reason=='edge_of_core_or_external_thermal_support'

def test_date_planning_requires_qualifying_scenes(tmp_path):
    csv=tmp_path/'scenes.csv';out=tmp_path/'plan.csv'
    pd.DataFrame([dict(scene_id='L8_fake_example',acquisition_utc='2025-07-11T15:40:00Z',
                       qualified_30m_grid_cells=7),
       dict(scene_id='L9_bad_example',acquisition_utc='2025-07-12T15:40:00Z',
            qualified_30m_grid_cells=1)]).to_csv(csv,index=False)
    rows=plan(csv,out,'2025-06-20','2025-06-23T23:00:00')
    assert len(rows)==1 and rows[0]['weather_start_utc_date']=='2025-07-09'
    assert rows[0]['weather_end_utc_date']=='2025-07-11'
    assert '48' in rows[0]['simulation_command']


def test_real_geometry_reconstruction_count():
    scene,sha,core,polys,ids=reconstructed_scene()
    assert len(scene['area'])==2082
    assert scene['building_count']==88
    assert len(sha)==64 and core.area>0 and len(polys)==len(ids)
    assert np.all(np.isin(scene['kind'][ids],['roof','ground']))


def test_time_alignment_matches_real_mesh_shape():
    scene,_,_,_,_=reconstructed_scene()
    n=len(scene['area'])
    d=pd.DataFrame({'time_utc':pd.to_datetime(['2025-06-20T15:00Z']*n+
        ['2025-06-20T16:00Z']*n,utc=True),
        'face_id':np.tile(np.arange(n),2),
        'surface_temp_C_simulated':np.repeat([25.,35.],n),
        'air_temp_C_grid':np.repeat([20.,24.],n),
        'area_m2':np.tile(scene['area'],2),
        'x_m':np.tile(scene['center'][:,0],2),
        'y_m':np.tile(scene['center'][:,1],2)})
    interpolated,error=interpolate_faces(d,'2025-06-20T15:30:00Z',scene)
    assert error is None
    T,air,t0,t1=interpolated
    assert len(T)==2082 and np.allclose(T,30) and air==22
    assert interpolate_faces(d,'2025-06-25T15:00:00Z',scene)[1]=='time_outside_model_range'

def test_optional_heldout_refuses_small_sample(tmp_path):
    from evaluate_scene_holdout import evaluate
    file=tmp_path/'tiny.csv'
    pd.DataFrame([{'scene_id':str(i),'acquisition_utc':f'2025-06-{i+1:02d}T15:40Z',
                   'observed_scene_mean_C':30+i,'modeled_scene_mean_proxy_C':30+i}
                  for i in range(3)]).to_csv(file,index=False)
    with pytest.raises(ValueError,match='Need >=6'):
        evaluate(file,tmp_path/'out')

def test_optional_heldout_never_uses_test_dates_for_fit(tmp_path):
    from evaluate_scene_holdout import evaluate
    file=tmp_path/'six_dates.csv'
    pd.DataFrame([{'scene_id':str(i),'acquisition_utc':f'2025-06-{i+1:02d}T15:40Z',
                   'observed_scene_mean_C':32+i if i<3 else 45+i,
                   'modeled_scene_mean_proxy_C':30+i}
                  for i in range(6)]).to_csv(file,index=False)
    r=evaluate(file,tmp_path/'out')
    assert r['calibration_additive_bias_C']==2
    assert len(r['heldout_dates'])==3
    assert r['heldout_raw']['MAE_C']==15
    assert r['heldout_bias_corrected']['MAE_C']==13
