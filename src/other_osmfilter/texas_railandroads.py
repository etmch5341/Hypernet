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
OUTPUT_PATH = "data/output/TX"
FILE_NAME = "texas-latest.osm.pbf"

if __name__=='__main__':
    
    if sys.version_info.major == 2:
        raise RuntimeError('Unsupported python version')
    
    # Defining Filters
    prefilter = {Node: {}, Way: {"highway":["primary",],}, Relation: {}}
    blackfilter = [(("highway", "primary"), ("destination", "Houston"))]
    whitefilter =[(("highway","primary"), ("destination", "Austin"))]
    
    # TODO: First get some type of viewable linestring structure like all roads from cedar park to austin (so just a smaller scale version)
    # TODO: Then fix filter so that I get all highways/roads that lead into Austin/Dallas
    
    print(type(whitefilter))
    
    # json file for the Data dictionary
    JSON_outputfile = os.path.join(os.getcwd(),OUTPUT_PATH+FILE_NAME)
    
    # json file for the Elements dictionary is automatically written to 'tests/output/LI/Elements'+filename)
    PBF_inputfile = os.path.join(os.getcwd(), INPUT_PATH+FILE_NAME)
    
    #TODO: Is there a way to have multiple elements be included 
    
    # TODO: is it possible to have multiple elements?
    # Extract Data and Elements
    element_name = "primary"
    [Data,Elements]=run_filter(element_name, 
                               PBF_inputfile, 
                               JSON_outputfile, 
                               prefilter,whitefilter,blackfilter, 
                               NewPreFilterData=True, 
                               CreateElements=True, 
                               LoadElements=False, 
                               verbose=True, 
                               multiprocess=True)
        
    # Export GeoJson
    #TODO: Test for elements output and 
    export_geojson(Elements[element_name]['Way'],Data,filename= OUTPUT_PATH + 'TX_RailandRoad.geojson',jsontype='Line')
    
    