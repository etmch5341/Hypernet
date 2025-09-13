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
OUTPUT_PATH = "data/output/SW-road/"
FILE_NAME = "switzerland-latest.osm.pbf"

if __name__=='__main__':
    
    # Check Python Version
    if sys.version_info.major == 2:
        raise RuntimeError('Unsupported python version')
    
    # Define the filters
    # prefilter = {
    #     Node: {},
    #     Way: {'highway': ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential'],
    #           'railway': ['rail', 'subway', 'tram']},
    #     Relation: {}
    # }
    # Road only
    prefilter = {
        Node: {},
        Way: {'highway': ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential']},
        Relation: {}
    }
    # Rail only
    # prefilter = {
    #     Node: {},
    #     Way: {'railway': ['rail', 'subway', 'tram']},
    #     Relation: {}
    # }

    # Exclude elements filter
    blackfilter = []

    # Specify the roadway types to retain
    # whitefilter = [
    #     (('railway', 'rail'),
    #      ('railway', 'subway'),
    #      ('railway', 'tram'))
    # ]
    # Road only
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

    # Run Filtering
    element_name = "filtered-road"
    [Data,Elements]=run_filter(element_name, 
                               PBF_inputfile, 
                               JSON_outputfile, 
                               prefilter,whitefilter,blackfilter, 
                               NewPreFilterData=True, 
                               CreateElements=True, 
                               LoadElements=False, 
                               verbose=True, 
                               multiprocess=True)
    
    # Check elements and data output
    # print(f"Elements: {Elements}")
    # print(f"Data: {Data}")
    
    # Debugging outputs to check the number of elements and tags
    print(f"Data for 'Way' elements: {len(Data['Way'])}")
    print(f"Elements after filtering: {len(Elements[element_name]['Way'])}")

    # Export GeoJson
    OUTPUT_NAME = 'SW_roads.geojson'    
    export_geojson(
        Elements[element_name]['Way'],
        Data,
        filename= OUTPUT_PATH + OUTPUT_NAME,
        jsontype='Line'
    )

    print(f"{OUTPUT_NAME} exported {OUTPUT_PATH}{OUTPUT_NAME}")