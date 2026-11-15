"""Optional surrounding-building buffer from NYC's actual footprint records.

Uses ONLY municipal polygons/heights; no synthetic building locations are added.
Results may be newer than any historical weather date. Preserve dataset URL,
query, time and per-building ID. Context is optional, not silently merged.
"""
from pathlib import Path
import argparse,sys,json,hashlib
from shapely.geometry import mapping
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from urbanray import fetch_nyc_buildings,CORE_BBOX

p=argparse.ArgumentParser()
p.add_argument('--context-deg',type=float,default=.002)
p.add_argument('--out',default='data/nyc_optional_context.geojson')
a=p.parse_args()
if not 0<a.context_deg<=.01:raise ValueError('context-deg out of supported range')
records,audit=fetch_nyc_buildings(bbox=CORE_BBOX,context_deg=a.context_deg)
features=[]
for rec in records:
    features.append({'type':'Feature','geometry':mapping(rec['geometry']),
       'properties':{'id':rec['id'],'height_m':rec['height_m'],
                     'height_roof_ft':rec['height_roof_ft'],
                     'source':'NYC Open Data official OTI footprint API; optional context'}})
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'type':'FeatureCollection','features':features}))
audit['sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
audit['warning']='2026 data != verified 2025 building conditions; extra geometry changes model boundary'
out.with_suffix('.provenance.json').write_text(json.dumps(audit,indent=2))
print('Wrote',len(features),'real municipal building polygons to',out)
