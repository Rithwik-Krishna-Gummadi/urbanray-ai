import sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from physics_plus import (longwave_exchange,build_reciprocal_operator,
                          step_two_layer,materials_with_roof_alpha,
                          thermal_properties,ASSUMED_MATERIALS,SIGMA)
from urbanray import synthetic_canyon,build_scene


def test_longwave_equal_temperature_black_surfaces():
    area=np.array([1.,2.])
    P=csr_matrix(np.array([[0.,.6],[.3,0.]]))
    sky=np.array([.4,.7])
    T=np.full(2,293.15)
    net,_=longwave_exchange(T,P,sky,area,np.ones(2),SIGMA*T[0]**4)
    assert np.max(np.abs(net))<1e-8


def test_longwave_equal_temperature_gray_surfaces():
    area=np.array([1.,2.])
    P=csr_matrix(np.array([[0.,.6],[.3,0.]]))
    T=np.full(2,295.)
    net,_=longwave_exchange(T,P,np.array([.4,.7]),area,np.array([.85,.95]),SIGMA*295.**4)
    assert np.max(np.abs(net))<1e-5


def test_longwave_conservation_closed_two_surface():
    area=np.array([2.,2.]); P=csr_matrix([[0.,1.],[1.,0.]])
    T=np.array([310.,290.]); eps=np.array([.9,.9])
    net,_=longwave_exchange(T,P,np.zeros(2),area,eps,200.)
    assert abs(float(np.dot(area,net)))<1e-7
    assert net[0]<0 and net[1]>0


def test_material_physical_and_assumptions():
    assert materials_with_roof_alpha(.58)['roof'].alpha==.58
    kinds=np.array(['roof','wall','ground'])
    p=thermal_properties(kinds)
    assert np.all(p['Cs']>0) and np.all(p['Cd']>0)


def test_geometric_operator_reciprocity_and_energy():
    scene=build_scene(synthetic_canyon(),max_buildings=3,ground_cell_m=45)
    P,sky,md=build_reciprocal_operator(scene,rays=3,seed=45)
    area=scene['area']
    assert np.all((sky>=0)&(sky<=1))
    assert np.allclose(np.asarray(P.sum(axis=1)).ravel()+sky,1,atol=1e-9)
    Q=P.multiply(area[:,None]); diff=Q-Q.T
    assert np.max(np.abs(diff.data))<1e-6 if diff.nnz else True
    assert md['num_patches']==len(area)


def test_two_layer_night_test_weather_no_explosion():
    scene=build_scene(synthetic_canyon(),max_buildings=3,ground_cell_m=45)
    P,sky,_=build_reciprocal_operator(scene,rays=2,seed=45)
    n=len(scene['area']);T=np.full(n,295.)
    Ts,Td,md=step_two_layer(T,T,np.zeros(n),295.,2.,65.,.1,P,sky,scene,dt=900,deep_reference_K=295.)
    assert np.isfinite(Ts).all() and (Ts>260).all() and (Ts<330).all()
    assert md['max_surface_residual_Wm2']<0.3


def test_longwave_rejects_bad_temperature():
    with pytest.raises(ValueError):
        longwave_exchange(np.array([-1.]),csr_matrix([[0.]]),np.array([1.]),
                          np.array([1.]),np.array([.9]),250.)

def test_shortwave_energy_model_closed_and_open():
    from physics_plus import shortwave_from_operator
    P=csr_matrix(np.array([[0.,.4],[.2,0.]]))
    area=np.array([1.,2.]); sky=np.array([.6,.8])
    SW,meta=shortwave_from_operator(np.array([100.,0.]),P,sky,area,np.array([.4,.5]),3)
    assert SW.shape==(2,)
    assert meta['relative_residual']<1e-12
    assert meta['absorbed_W']+meta['escaped_W']+meta['unresolved_W']==pytest.approx(100.)
    SW2,_=shortwave_from_operator(np.array([100.,0.]),P,sky,area,np.zeros(2),4)
    assert SW2[0]==pytest.approx(100.)
