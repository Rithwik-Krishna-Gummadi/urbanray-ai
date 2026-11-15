"""Fetch hourly *gridded reanalysis* from documented Open-Meteo Archive API.

This cannot verify measured rooftop weather. Saves JSON response + request URL +
SHA-256 so an independent reviewer can repeat the same request. UTC avoids DST gaps.
"""
import argparse,requests,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd

HOURLY=['temperature_2m','relative_humidity_2m','wind_speed_10m',
        'wind_direction_10m','cloud_cover','direct_normal_irradiance_instant',
        'diffuse_radiation_instant']
BASE='https://archive-api.open-meteo.com/v1/archive'


def download(start,end,out,latitude=40.753,longitude=-73.9867):
    dates=(pd.Timestamp(end)-pd.Timestamp(start)).days+1
    if dates<1 or dates>366:raise ValueError('Invalid range: 1-366 days')
    params={'latitude':latitude,'longitude':longitude,
        'start_date':start,'end_date':end,'timezone':'UTC',
        'wind_speed_unit':'ms','hourly':','.join(HOURLY)}
    response=requests.get(BASE,params=params,timeout=120)
    response.raise_for_status()
    payload=response.json()
    if 'hourly' not in payload or 'time' not in payload['hourly']:
        raise ValueError('Historical API did not return hourly weather: '+str(payload)[:300])
    hourly=payload['hourly']
    if any(k not in hourly for k in HOURLY):raise ValueError('Missing requested variables')
    if len(hourly['time']) != dates*24:raise ValueError('Unexpected number of UTC hourly records')
    df=pd.DataFrame(hourly)
    df['time_local']=pd.to_datetime(df['time'],utc=True).dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
    df=df.drop(columns=['time'])
    if df[HOURLY].isna().any().any():raise ValueError('Missing gridded weather samples; refuse to invent values')
    assert (pd.to_datetime(df['time_local'],utc=True).diff().dropna()==pd.Timedelta(hours=1)).all()
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False)
    audit={'source_url':response.url,'response_sha256':hashlib.sha256(response.content).hexdigest(),
           'csv_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
           'source_type':'gridded modeled/reanalysis weather; NOT ground-station measurements',
           'units':'temperature Celsius, wind meters/second, irradiance W/m2, humidity percent',
           'latitude_requested':latitude,'longitude_requested':longitude,
           'record_count':len(df),'UTC_first':df.time_local.iloc[0],
           'UTC_last':df.time_local.iloc[-1],
           'hourly_fields':HOURLY}
    out.with_suffix('.provenance.json').write_text(json.dumps(audit,indent=2))
    out.with_suffix('.api.json').write_text(json.dumps(payload))
    print('Saved',len(df),'hours to',out,'\nData source:',response.url)
    return df,audit

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--start',default='2025-06-18')
    p.add_argument('--end',default='2025-06-25')
    p.add_argument('--out',default='data/nyc_gridded_weather_2025_06_18_25.csv')
    p.add_argument('--latitude',type=float,default=40.753)
    p.add_argument('--longitude',type=float,default=-73.9867)
    a=p.parse_args()
    download(a.start,a.end,a.out,a.latitude,a.longitude)
