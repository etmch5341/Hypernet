
# Experimental Module to export Ways and Nodes to GeoJson
# %%
'''
Sample of use:
filename='test.geojson'
export_geojson(Elements['pipeline']['Way'],Data,filename,jsontype='Line')
filename2='test2.geojson'
export_geojson(Elements['pipeline']['Node'],Data,filename2,jsontype='Point')
''' 
import json
import logging

from . import osm_colors    as CC




def get_ref_properties(element):
    
    # for ref in element['tags']:
    #     print(ref)
    # print('--')
    #print(element['tags'])
    return element['tags']

#%%
def get_ref_coordinates(element,Data):
    #returns coordinates, and if a way is broken (broken_way) because not all nodes are stored in dict,
    #possibly due to applying a poly file to osm.pbf and cutting through ways
    longs=[]
    lats=[]
    coordinates=[]
    broken_way=False
    for ref in element['refs']:
        if Data['Node'].get(str(ref), 1000) != 1000:
            longs.append(Data['Node'][str(ref)]['lonlat'][0])
            lats.append(Data['Node'][str(ref)]['lonlat'][1])
        else :
            logging.warning(print(CC.Red+'Warning: Way:'+str(element['id'])+' removed, because OSM node '+str(ref)+' is missing!'+CC.End))
            broken_way=True

    for coords in zip(longs,lats):
        coordinates.append(list(coords))    
    return coordinates,broken_way

def get_coordinates(element):
    
    coordinates=element['lonlat']
    return coordinates

#%%

def create_GeoJson_Points(elements,Data):    
    features=[]
    for element in elements:
        features.append({'type'        : 'Feature',
                        'OSM-id'      : elements[element]['id'],
                        'geometry'    : 
                       {'type'        : 'Point',
                        'coordinates' : get_coordinates(elements[element])},
                        'properties'  : get_ref_properties(elements[element])},
                        
                        )
                                  
    return features  


def create_GeoJson_Lines(elements,Data):    
    features=[]
    for element in elements:
        feature={'type'        : 'Feature',
                 'OSM-id'      : elements[element]['id'],
                        'geometry'    : 
                       {'type'        : 'LineString',
                        'coordinates' : get_ref_coordinates(elements[element],Data)[0]},
                        'properties'  : get_ref_properties(elements[element])}                   
                        
        broken_way=get_ref_coordinates(elements[element],Data)[1]
        if broken_way==False:
            features.append(feature)
            
                        
                            
    return features

def export_geojson(Elements,Data,filename,jsontype='Line'):
    ### Elements=Elements['pipeline']['Way']
    if jsontype=='Point' or jsontype=='Node':
        collection=create_GeoJson_Points(Elements,Data)
        output={'type': 'FeatureCollection','features': collection}
        open(filename,"w",encoding="cp1252").write(json.dumps(output,indent=4))
    elif jsontype=='Line' or jsontype=='LineString':
        collection=create_GeoJson_Lines(Elements,Data)
        output={'type': 'FeatureCollection','features': collection}
        open(filename,"w",encoding="cp1252").write(json.dumps(output,indent=4))
    else:
        print('No such jsontype')
    



    
    
