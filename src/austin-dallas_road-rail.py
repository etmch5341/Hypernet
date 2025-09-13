import os
import urllib.request
import subprocess
import sys
import logging
from esy.osmfilter import run_filter, Node, Way, Relation, export_geojson

# Set up logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# === PATH SETUP ===
INPUT_PATH = "data/input/"
OUTPUT_PATH = "data/output/TX/"
os.makedirs(INPUT_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

FULL_PBF = os.path.join(INPUT_PATH, "texas-latest.osm.pbf")
CLIPPED_PBF = os.path.join(INPUT_PATH, "austin_dallas.osm.pbf")
OUTPUT_JSON = os.path.join(OUTPUT_PATH, "austin_dallas.osm.json")
OUTPUT_GEOJSON = os.path.join(OUTPUT_PATH, "austin_dallas.geojson")

# === DOWNLOAD FULL TEXAS FILE ===
if not os.path.exists(FULL_PBF):
    print("Downloading Texas .osm.pbf...")
    url = "https://download.geofabrik.de/north-america/us/texas-latest.osm.pbf"
    urllib.request.urlretrieve(url, FULL_PBF)
    print("Download complete:", FULL_PBF)

# === CLIP TO BBOX: (30, -98) -> (33, -96) ===
print("Clipping to Austin–Dallas corridor...")
bbox = "-98.0,30.0,-96.0,33.0"
subprocess.run([
    "osmium", "extract",
    "-b", bbox,
    FULL_PBF,
    "-o", CLIPPED_PBF
], check=True)
print("Clipping complete:", CLIPPED_PBF)

# === OSMFILTER: Setup Filters ===
prefilter = {
    Node: {},
    Way: {
        'highway': ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential'],
        'railway': ['rail', 'light_rail', 'subway', 'tram']
    },
    Relation: {}
}

whitefilter = [(
    ('highway', 'motorway'),
    ('highway', 'trunk'),
    ('highway', 'primary'),
    ('highway', 'secondary'),
    ('highway', 'tertiary'),
    ('highway', 'unclassified'),
    ('highway', 'residential'),
    ('railway', 'rail'),
    ('railway', 'light_rail'),
    ('railway', 'subway'),
    ('railway', 'tram')
)]

blackfilter = []

# === RUN OSMFILTER ===
element_name = "filtered-austin-dallas"
print("Running filter...")
Data, Elements = run_filter(
    element_name,
    CLIPPED_PBF,
    OUTPUT_JSON,
    prefilter,
    whitefilter,
    blackfilter,
    NewPreFilterData=True,
    CreateElements=True,
    LoadElements=False,
    verbose=True,
    multiprocess=False
)

# === EXPORT TO GEOJSON ===
print(f"Filtered {len(Elements[element_name]['Way'])} ways.")
export_geojson(
    Elements[element_name]['Way'],
    Data,
    filename=OUTPUT_GEOJSON,
    jsontype='Line'
)
print("GeoJSON export complete:", OUTPUT_GEOJSON)
