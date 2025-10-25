import os, sys, time
import logging
import urllib.request


from esy.osmfilter import  osm_colors as CC
from esy.osmfilter import run_filter, Node,Way,Relation 
from esy.osmfilter import export_geojson
from contextlib import contextmanager


logging.basicConfig()
logger=logging.getLogger(__name__)
#logger.setLevel(logging.INFO)
logger.setLevel(logging.WARNING)
    

if __name__=='__main__':
    
    if sys.version_info.major == 2:
        raise RuntimeError('Unsupported python version')
        
    #-------------------------------------------------------------------------------------------------
    #                About reducing the size of a pbf file
    # ------------------------------------------------------------------------------------------------
    # 
    # please notice, you can use Osmosis with a pbf and and a poly file to create a reduced pbf file
    # with the followin Syntax
    #
    # osmosis --read-pbf file="France.osm.pbf"--bounding-polygon file="Paris.poly" 
    #       completeWays=yes clipIncompleteEntities=true --write-pbf file="Paris.osm.pbf"
    #-------------------------------------------------------------------------------------------------
    
    
    #see docu for filter structures
    prefilter = {Node: {}, Way: {"man_made":["pipeline",],}, Relation: {}}
    blackfilter = [("pipeline","substation"),]
    whitefilter =[[("man_made","pipeline"),("waterway","drain")],]
    #paths to input and output
    
    #filename='liechtenstein-191101.osm.pbf'
    #filename='latvia-latest.osm.pbf'
    filename='portugal-latest.osm.pbf'
    # filename = "./data/texas-latest.osm.pbf"


    # json file for the Data dictionary
    # JSON_outputfile = os.path.join(os.getcwd(),'./tests/outputs/'+"test.geojson")
    JSON_outputfile = "./tests/outputs/test.geojson"
    # json file for the Elements dictionary is automatically written to 'tests/output/LI/Elements'+filename)
    
    # PBF_inputfile = os.path.join(os.getcwd(),'./tests/input/'+filename)
    PBF_inputfile = "./tests/inputs/" + filename
    
    # if not os.path.exists('tests/input/'+filename):
    #     print("here")
    #     filename, headers = urllib.request.urlretrieve(
    #     'https://download.geofabrik.de/europe/'+filename,
    #     filename='tests/input/'+filename
    #     )
    #     PBF_inputfile = filename

    elementname='pipeline'
    #see docu for data- and element-dictionary structures
    [Data,Elements]=run_filter(elementname,PBF_inputfile, JSON_outputfile,prefilter,whitefilter,blackfilter, 
                                NewPreFilterData=True, CreateElements=True, LoadElements=True,verbose=True,multiprocess=False)
    #print(CC.Cyan+'Elements:'+CC.End)
    #print(Elements)
    #At this point we could export the elements to a geojson-file
    export_geojson(Elements[elementname]['Way'],Data,filename='./tests/output/test.geojson',jsontype='Line')
    # You can also convert extract node elements
    # export_geojson(Elements[elementname]['Node'],Data,filename='test.geojson',jsontype='Point')
    

