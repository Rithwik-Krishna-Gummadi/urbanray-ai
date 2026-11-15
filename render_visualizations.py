"""Interactive HTML from finished physics+ results; titles disclose model origin."""
import argparse,sys,json
from pathlib import Path
import numpy as np,pandas as pd
import plotly.graph_objects as go
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from integrated88 import load_verified_geometry
from urbanray import build_scene

p=argparse.ArgumentParser()
p.add_argument('--results',default='outputs/physics_plus')
a=p.parse_args()
folder=Path(a.results)
frame=pd.read_csv(folder/'thermal_hourly_summary.csv')
meta=json.loads((folder/'audit.json').read_text())
subtitle='GRIDDed MODELED WEATHER; ASSUMED MATERIALS' if meta.get('weather_provenance_verified') else 'UNVERIFIED WEATHER — SMOKE/DEVELOPMENT ONLY'
fig=go.Figure()
for label,g in frame.groupby('scenario'):
    fig.add_scatter(x=g.time_utc,y=g.roof_C_simulated,mode='lines+markers',name=label)
fig.update_layout(title='UrbanRay: mean modeled roof temperatures — '+subtitle,
    xaxis_title='Timestamp (UTC)',yaxis_title='Simulated roof temperature (°C)',height=530)
fig.write_html(folder/'roof_temperature_comparison.html',include_plotlyjs='cdn')

faces_path=folder/'physics_training_faces.csv'
if faces_path.exists():
    face=pd.read_csv(faces_path)
    records,_=load_verified_geometry(ROOT/'data'/'nyc_88_source.geojson')
    scene=build_scene(records,ground_cell_m=25,max_buildings=88)
    labels=sorted(face.scenario.unique())
    chosen=[x for x in ('dark_roof','reflective_roof') if x in labels]
    for scenario in chosen:
        times=face[face.scenario==scenario].time_utc.unique()
        g=face[(face.scenario==scenario)&(face.time_utc==times[-1])].sort_values('face_id')
        if len(g)!=len(scene['area']):
            print('Skipping geometry figure: face count differs (possibly buffer geometry)')
            continue
        verts=scene['triangles'].reshape(-1,3); ind=np.arange(len(verts)).reshape(-1,3)
        fig3=go.Figure(go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],
            i=ind[:,0],j=ind[:,1],k=ind[:,2],
            intensity=g.surface_temp_C_simulated.to_numpy(),intensitymode='cell',
            colorscale='Turbo',flatshading=True,
            colorbar={'title':'Modeled °C'},cmin=10,cmax=75))
        fig3.update_layout(title=f'UrbanRay {scenario} — 3D MODEL temperatures, {subtitle}',
            scene=dict(aspectmode='data',xaxis_title='Easting (m)',
                yaxis_title='Northing (m)',zaxis_title='Height (m)'),
            height=720,margin=dict(l=0,r=0,b=0,t=60))
        fig3.write_html(folder/f'3D_{scenario}.html',include_plotlyjs='cdn')
print('Visualization files written to',folder)
