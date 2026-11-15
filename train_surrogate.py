"""Leakage-aware ML benchmark against the REAL 88-building physics simulation.

Requires multiple days of source-tagged simulated face data. Cannot and will
not claim validation against measured NYC temperatures: these are model labels.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import joblib

CATEGORICAL=['kind_code']
WEATHER=['DNI_grid_Wm2','DHI_grid_Wm2','air_temp_C_grid','wind_grid_ms',
         'sun_east','sun_north','sun_up']
BASIC=['area_m2','z_m','albedo_assumed','kind_code','source_flux_Wm2',
       'direct_plus_sky_abs_Wm2']+WEATHER
PHYSICS=BASIC+['sky_fraction_modeled','nx','ny','nz']

def load_training(path):
    df=pd.read_csv(path)
    needed={'time_utc','date_utc','scenario','face_id','building_id','kind',
            'x_m','y_m','albedo_assumed','multibounce_abs_Wm2','surface_temp_C_simulated'}|(set(PHYSICS)-{'kind_code'})
    if not needed.issubset(df):raise ValueError('Dataset missing fields: '+str(sorted(needed-set(df))))
    df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=list(needed)).copy()
    df['kind_code']=df['kind'].map({'roof':0,'wall':1,'ground':2}).astype(int)
    df['date_utc']=pd.to_datetime(df['time_utc'],utc=True).dt.strftime('%Y-%m-%d')
    # Spatial squares; geographic distance in local projected meters.
    df['block_x']=np.floor(df.x_m/110).astype(int)
    df['block_y']=np.floor(df.y_m/110).astype(int)
    df['spatial_block']=df.block_x.astype(str)+'_'+df.block_y.astype(str)
    return df

def split(df,seed=62):
    dates=sorted(df.date_utc.unique())
    if len(dates)<4:raise ValueError('Need >=4 independent UTC simulation dates for an honest train/val/test split. Got '+str(len(dates)))
    blocks=sorted(df.spatial_block.unique())
    if len(blocks)<5:raise ValueError('Need >=5 spatial blocks, got '+str(len(blocks)))
    # Holdout entire blocks for all dates: no training point shares a test block.
    holdout={b for b in blocks if int(hashlib.sha256((str(seed)+b).encode()).hexdigest()[:8],16)%5==0}
    if not holdout:holdout={blocks[-1]}
    inblock=~df.spatial_block.isin(holdout)
    train=df[inblock&df.date_utc.isin(dates[:-2])].copy()
    val=df[inblock&(df.date_utc==dates[-2])].copy()
    test=df[df.spatial_block.isin(holdout)&(df.date_utc==dates[-1])].copy()
    if min(len(train),len(val),len(test))<30:raise ValueError('Insufficient train/validation/test faces with full blocked split')
    return train,val,test,{'dates_train':dates[:-2], 'date_validation':dates[-2],
         'date_test':dates[-1], 'test_spatial_blocks':sorted(holdout),
         'train_rows':len(train),'val_rows':len(val),'test_rows':len(test)}

def fit_and_score(train,val,test,output,seed=62):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target='multibounce_abs_Wm2'
    original_lengths={'train':len(train),'validation':len(val),'test':len(test)}
    # Bound training cost; seeded subsampling stays inside the original
    # temporal/spatial groups and never imports withheld rows.
    if len(train)>45000: train=train.sample(n=45000,random_state=seed).copy()
    if len(val)>15000: val=val.sample(n=15000,random_state=seed+1).copy()
    # Selection on validation only. Never select based on the test set.
    candidates=[{'max_leaf_nodes':15,'learning_rate':.07,'max_iter':150},
                {'max_leaf_nodes':31,'learning_rate':.065,'max_iter':200},
                {'max_leaf_nodes':20,'learning_rate':.05,'max_iter':250}]
    trained=[]
    for params in candidates:
        model=HistGradientBoostingRegressor(**params,random_state=seed,
              l2_regularization=1.0,min_samples_leaf=20)
        model.fit(train[PHYSICS],train[target])
        score=mean_absolute_error(val[target],model.predict(val[PHYSICS]))
        trained.append((score,model,params))
    trained.sort(key=lambda x:x[0]);_,_,params=trained[0]
    final=HistGradientBoostingRegressor(**params,random_state=seed,
              l2_regularization=1.0,min_samples_leaf=20)
    combined=pd.concat([train,val],ignore_index=True)
    final.fit(combined[PHYSICS],combined[target])
    y=test[target].to_numpy()
    preds={
      'Mean-only':np.full(len(test),float(train[target].mean())),
      'No-reflection physics':test.direct_plus_sky_abs_Wm2.to_numpy(),
      'Physics-informed ML':np.maximum(0.,final.predict(test[PHYSICS]))
    }
    rows=[]
    for name,p in preds.items():
        rows.append({'model':name,'test_MAE_Wm2':float(mean_absolute_error(y,p)),
                     'test_RMSE_Wm2':float(np.sqrt(mean_squared_error(y,p))),
                     'test_R2':float(r2_score(y,p))})
    metrics=pd.DataFrame(rows)
    metrics.to_csv(output/'heldout_radiation_scores.csv',index=False)
    predfile=test[['time_utc','date_utc','scenario','building_id','kind','face_id','spatial_block']].copy()
    predfile['physics_label_Wm2']=y
    for name,p in preds.items():predfile['pred_'+name.replace(' ','_')]=p
    predfile.to_csv(output/'heldout_predictions.csv',index=False)
    joblib.dump({'model':final,'features':PHYSICS,'target':target,'note':'Targets from simulator, not measured truth'},output/'physics_surrogate.joblib')
    audit={'label':'SIMULATED multi-bounce shortwave absorption, not measured temperature',
           'original_partition_sizes':original_lengths,
           'fit_rows_after_seeded_sampling':int(len(train)+len(val)),
           'validation_rows_after_seeded_sampling':len(val),
           'test_rows_all_heldout':len(test),
           'selection':'validation only, refit on train+validation',
           'best_validation_MAE_Wm2':float(trained[0][0]),'chosen_hyperparameters':params,
           'results':rows,'importance':'No statements about improvement unless test metric supports it'}
    (output/'ml_audit.json').write_text(json.dumps(audit,indent=2))
    return metrics,audit

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--faces',required=True)
    p.add_argument('--out',default='outputs/ml_surrogate')
    a=p.parse_args()
    df=load_training(a.faces)
    train,val,test,audit=split(df)
    metrics,sc=fit_and_score(train,val,test,a.out)
    (Path(a.out)/'split_audit.json').write_text(json.dumps(audit,indent=2,default=str))
    print(metrics.to_string(index=False))
    print('CAUTION: AI was tested against synthetic-physics labels, not city measurements')
