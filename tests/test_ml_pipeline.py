import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_surrogate import load_training,split,fit_and_score

def test_temporal_spatial_holdout_pipeline(tmp_path):
    # Synthetic DATA used solely for end-to-end software testing (not research results).
    rng=np.random.default_rng(1)
    rows=[]
    for d in range(5):
        for bi in range(10):
            for idx in range(24):
                x=bi*111.; y=idx%4*113.
                sky=.3+.55*rng.random(); src=150+700*rng.random()
                albedo=.18 if d%2==0 else .58
                direct=src*(1-albedo)
                flux=direct+15*sky+5*rng.normal()
                rows.append({'time_utc':f'2025-07-{15+d:02d}T12:00:00+00:00',
                    'date_utc':f'2025-07-{15+d:02d}', 'scenario':'test',
                    'face_id':bi*24+idx,'building_id':f'BUILD_{bi}',
                    'kind':['roof','wall','ground'][idx%3],
                    'x_m':x,'y_m':y,'z_m':15.,'area_m2':8.,
                    'albedo_assumed':albedo,'source_flux_Wm2':src,
                    'direct_plus_sky_abs_Wm2':direct,'multibounce_abs_Wm2':flux,
                    'surface_temp_C_simulated':30+flux/100,
                    'DNI_grid_Wm2':750,'DHI_grid_Wm2':110,
                    'air_temp_C_grid':29,'wind_grid_ms':3,
                    'sun_east':.2,'sun_north':-.5,'sun_up':.8,
                    'nx':0,'ny':0,'nz':1,'sky_fraction_modeled':sky})
    csv=tmp_path/'synthetic.csv';pd.DataFrame(rows).to_csv(csv,index=False)
    df=load_training(csv)
    a,b,c,meta=split(df)
    assert set(a.spatial_block).isdisjoint(set(c.spatial_block))
    assert set(a.date_utc).isdisjoint(set(c.date_utc))
    scores,_=fit_and_score(a,b,c,tmp_path/'result')
    assert len(scores)==3 and np.isfinite(scores.test_MAE_Wm2).all()
