// UrbanRay — Find many observed Landsat dates for the NYC 88-building core.
// Google Earth Engine Code Editor: https://code.earthengine.google.com/
// Exports an ALL-scenes inventory, a shortlist of observed dates, and
// the quality-screened pixel observations for those shortlisted scenes.
// Every shortlisted satellite date still requires an UrbanRay simulation.

var REGION = ee.Geometry.Rectangle(
  [-73.98775, 40.7521, -73.98565, 40.7539], null, false
);
var START = '2025-01-01';
var END_EXCLUSIVE = '2026-11-16';
var MAX_UNCERTAINTY_K = 5.0; // exploratory screening, not 5 K model accuracy
var MIN_CLOUD_DISTANCE_KM = 0.0;
var MIN_VALID_GRID_CELLS = 10; // 30m output cells, NOT independent thermal samples
var DRIVE_FOLDER = 'UrbanRayValidation';

var scenes = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
  .filterBounds(REGION)
  .filterDate(START, END_EXCLUSIVE)
  .filter(ee.Filter.eq('PROCESSING_LEVEL', 'L2SP'))
  .sort('system:time_start');

// Convert the projection's WKT affine matrix to the six coefficients
// expected by the existing UrbanRay validation script.
function affineValue(wkt, key) {
  var pattern = '"' + key + '"\\s*,\\s*([-+0-9.eE]+)';
  var match = ee.List(ee.String(wkt).match(pattern));
  return ee.String(match.cat(ee.List(['0', '0'])).get(1));
}
function affineFor(projection) {
  var wkt = ee.String(projection.transform());
  var keys = ['elt_0_0', 'elt_0_1', 'elt_0_2',
              'elt_1_0', 'elt_1_1', 'elt_1_2'];
  return ee.List(keys.map(function(key) {
    return affineValue(wkt, key);
  })).join(';');
}

function eligiblePixels(image) {
  var img = ee.Image(image);
  var st = img.select('ST_B10');
  var qa = img.select('QA_PIXEL');
  var sat = img.select('QA_RADSAT');
  var uncert = img.select('ST_QA').multiply(0.01)
    .rename('st_uncertainty_k');
  var cloudDist = img.select('ST_CDIST').multiply(0.01)
    .rename('cloud_distance_km');
  var valid = st.gt(0)
    .and(qa.bitwiseAnd(191).eq(0))
    .and(sat.eq(0))
    .and(uncert.lte(MAX_UNCERTAINTY_K))
    .and(cloudDist.gte(MIN_CLOUD_DISTANCE_KM));
  return ee.Image.cat([
    st.multiply(0.00341802).add(149).subtract(273.15)
      .rename('lst_c'),
    uncert,
    cloudDist,
    img.select('ST_EMIS').multiply(0.0001)
      .rename('retrieval_emissivity'),
    qa.rename('qa_pixel'),
    sat.rename('qa_radsat'),
    ee.Image.pixelLonLat().rename(['center_lon', 'center_lat'])
  ]).updateMask(valid);
}
function sampleScene(image) {
  var img = ee.Image(image);
  return eligiblePixels(img).sample({
    region: REGION,
    projection: img.select('ST_B10').projection(),
    tileScale: 4,
    dropNulls: true,
    geometries: false
  });
}

function summarizeScene(image) {
  var img = ee.Image(image);
  var sampled = sampleScene(img);
  var n = sampled.size();
  var projection = img.select('ST_B10').projection();
  var localDay = img.date().format('YYYY-MM-dd', 'America/New_York');
  return ee.Feature(null, {
    scene_id: img.get('LANDSAT_PRODUCT_ID'),
    acquisition_utc: img.date().format("YYYY-MM-dd'T'HH:mm:ss'Z'", 'UTC'),
    acquisition_local_date: localDay,
    sensor: img.get('SPACECRAFT_ID'),
    scene_cloud_cover_pct: img.get('CLOUD_COVER'),
    processing_level: img.get('PROCESSING_LEVEL'),
    qualified_30m_grid_cells: n,
    qualified_grid_mean_lst_c: ee.Algorithms.If(
      n.gt(0), sampled.aggregate_mean('lst_c'), null),
    mean_st_uncertainty_k: ee.Algorithms.If(
      n.gt(0), sampled.aggregate_mean('st_uncertainty_k'), null),
    projection_crs: projection.crs(),
    projection_affine: affineFor(projection)
  });
}

var allScenes = ee.FeatureCollection(
  scenes.toList(scenes.size()).map(function(im) {
    return summarizeScene(ee.Image(im));
  })
);
var goodScenes = allScenes
  .filter(ee.Filter.gte('qualified_30m_grid_cells', MIN_VALID_GRID_CELLS))
  .sort('qualified_30m_grid_cells', false);

var goodIds = goodScenes.aggregate_array('scene_id');
var goodImageCollection = scenes.filter(
  ee.Filter.inList('LANDSAT_PRODUCT_ID', goodIds)
);
function pixelRecords(image) {
  var img = ee.Image(image);
  var projection = img.select('ST_B10').projection();
  var props = ee.Dictionary({
    scene_id: img.get('LANDSAT_PRODUCT_ID'),
    acquisition_utc: img.date().format("YYYY-MM-dd'T'HH:mm:ss'Z'", 'UTC'),
    sensor: img.get('SPACECRAFT_ID'),
    scene_cloud_cover_pct: img.get('CLOUD_COVER'),
    processing_level: img.get('PROCESSING_LEVEL'),
    projection_crs: projection.crs(),
    projection_affine: affineFor(projection),
    sampling_grid: 'native_30m_ST_B10_grid',
    source_dataset: 'USGS_Landsat_C2_T1_L2',
    st_qa_limit_k: MAX_UNCERTAINTY_K,
    min_cloud_distance_km: MIN_CLOUD_DISTANCE_KM
  });
  return sampleScene(img).map(function(f) {
    return ee.Feature(null, ee.Feature(f).toDictionary().combine(props));
  });
}
var pixelTable = ee.FeatureCollection(
  goodImageCollection.toList(goodImageCollection.size()).map(function(im) {
    return pixelRecords(ee.Image(im));
  })
).flatten();

print('Total Landsat candidate scenes:', scenes.size());
print('Scenes meeting quality and minimum-cell filter:', goodScenes.size());
print('Best dates (more 30m cells first):', goodScenes.limit(30));
print('First five exported pixel rows:', pixelTable.limit(5));
Map.centerObject(REGION, 16);
Map.addLayer(REGION, {color: 'yellow'}, 'UrbanRay 88-building study core');

Export.table.toDrive({
  collection: allScenes,
  description: 'UrbanRay_All_Scene_Inventory_2025_2026',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'urbanray_all_scene_inventory_2025_2026',
  fileFormat: 'CSV',
  selectors: ['scene_id', 'acquisition_utc', 'acquisition_local_date',
    'sensor', 'scene_cloud_cover_pct', 'processing_level',
    'qualified_30m_grid_cells', 'qualified_grid_mean_lst_c',
    'mean_st_uncertainty_k', 'projection_crs', 'projection_affine']
});
Export.table.toDrive({
  collection: goodScenes,
  description: 'UrbanRay_Qualified_Matching_Dates_2025_2026',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'urbanray_qualified_matching_dates_2025_2026',
  fileFormat: 'CSV',
  selectors: ['scene_id', 'acquisition_utc', 'acquisition_local_date',
    'sensor', 'qualified_30m_grid_cells', 'qualified_grid_mean_lst_c',
    'mean_st_uncertainty_k', 'scene_cloud_cover_pct',
    'projection_crs', 'projection_affine']
});
Export.table.toDrive({
  collection: pixelTable,
  description: 'UrbanRay_Qualified_Landsat_Pixels_2025_2026',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'urbanray_qualified_landsat_pixels_2025_2026',
  fileFormat: 'CSV',
  selectors: ['scene_id', 'acquisition_utc', 'sensor',
    'scene_cloud_cover_pct', 'processing_level',
    'projection_crs', 'projection_affine', 'sampling_grid',
    'source_dataset', 'st_qa_limit_k', 'min_cloud_distance_km',
    'center_lon', 'center_lat', 'lst_c', 'st_uncertainty_k',
    'cloud_distance_km', 'retrieval_emissivity', 'qa_pixel',
    'qa_radsat']
});
print('Start all THREE CSV export tasks in the Tasks tab.');
print('The shortlist is OBSERVATION DATES, not yet model-matched dates.');
