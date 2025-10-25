import os, sys, time
import logging
import urllib.request


from esy.osmfilter import osm_colors as CC
from esy.osmfilter import run_filter, Node,Way,Relation 
from esy.osmfilter import export_geojson
from contextlib import contextmanager


logging.basicConfig()
logger=logging.getLogger(__name__)
#logger.setLevel(logging.INFO)
logger.setLevel(logging.WARNING)

# File paths
INPUT_PATH = "data/input/"
OUTPUT_PATH = "data/output/LI/"
FILE_NAME = "liechtenstein-191101.osm.pbf"
# INPUT_PATH = "data/input/"
# OUTPUT_PATH = "data/output/TX"
# FILE_NAME = "texas-latest.osm.pbf"

if __name__=='__main__':
    
    if sys.version_info.major == 2:
        raise RuntimeError('Unsupported python version')
    
    # Define the filters
    # prefilter = {
    #     Node: {},
    #     Way: {'highway': []},  # Accept all elements with the `highway` tag
    #     Relation: {}
    # }
    prefilter = {
        Node: {},
        Way: {'highway': ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential']},
        Relation: {}
    }

    # Exclude any non-roadway or minor paths
    # blackfilter = [
    #     ('highway', 'cycleway'),
    #     ('highway', 'path'),
    #     ('highway', 'footway'),
    #     ('highway', 'pedestrian')
    # ]
    blackfilter = []

    # Specify the roadway types to retain
    # whitefilter = [(
    #     ('highway', 'motorway'),
    #     ('highway', 'primary'),
    #     ('highway', 'secondary'),
    #     ('highway', 'tertiary'),
    #     ('highway', 'residential'),
    #     ('highway', 'unclassified')
    # )]
    
    whitefilter = [(
        ('highway', 'motorway'),
        ('highway', 'trunk'),
        ('highway', 'primary'),
        ('highway', 'secondary'),
        ('highway', 'tertiary'),
        ('highway', 'unclassified'),
        ('highway', 'residential')
    )]
    
    
    
    # Json file for the Data dictionary
    JSON_outputfile = os.path.join(os.getcwd(),OUTPUT_PATH+FILE_NAME)
    
    # Json file for the Elements dictionary
    PBF_inputfile = os.path.join(os.getcwd(), INPUT_PATH+FILE_NAME)

    # # Run the prefilter phase
    # [Data, _] = run_filter(
    #     'roadways-prefilter',
    #     PBF_inputfile,
    #     JSON_outputfile,
    #     prefilter,
    #     whitefilter,
    #     blackfilter,
    #     NewPreFilterData=True,
    #     CreateElements=False,
    #     LoadElements=False,
    #     verbose=True,
    #     multiprocess=True
    # )
    
    # # Run the pre/main filter phase
    # [_, Elements] = run_filter(
    #     'filtered-roadways',
    #     PBF_inputfile,
    #     JSON_outputfile,
    #     prefilter,
    #     whitefilter,
    #     blackfilter,
    #     NewPreFilterData=True,
    #     CreateElements=True,
    #     LoadElements=False,
    #     verbose=True,
    #     multiprocess=True
    # )
    
    element_name = "filtered-roadways"
    [Data,Elements]=run_filter(element_name, 
                               PBF_inputfile, 
                               JSON_outputfile, 
                               prefilter,whitefilter,blackfilter, 
                               NewPreFilterData=True, 
                               CreateElements=True, 
                               LoadElements=False, 
                               verbose=True, 
                               multiprocess=False)
    
    # Check elements and data output
    # print(f"Elements: {Elements}")
    # print(f"Data: {Data}")
    
    # Debugging outputs to check the number of elements and tags
    print(f"Data for 'Way' elements: {len(Data['Way'])}")
    print(f"Elements after filtering: {len(Elements[element_name]['Way'])}")

        
    # Export GeoJson
    OUTPUT_NAME = 'LI_roads.geojson'
    # export_geojson(Elements[element_name]['Way'],Data,filename= OUTPUT_PATH + OUTPUT_NAME, jsontype='Line')
    
    # Export to GeoJSON for visualization
    export_geojson(
        Elements['filtered-roadways']['Way'],
        Data,
        filename= OUTPUT_PATH + OUTPUT_NAME,
        jsontype='Line'
    )

    print(f"Roadways exported {OUTPUT_PATH}{OUTPUT_NAME}")