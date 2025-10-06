import itertools
from   pathlib           import Path

from itertools import accumulate,starmap

from . import osm_colors    as CC

from esy.osm.pbf import osmformat_pb2, read_blob, Node, Way, Relation
from esy.osm.pbf.file import parse_tags, iter_blocks, decode_strmap

import sys 
import time 
import contextlib
import multiprocessing
import json
import os
import logging

logging.basicConfig()
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@contextlib.contextmanager
def benchmark(name):
    '''
    **Purpose**
        is used to calculate the execution time of the prefiltering process
    '''
    start = time.time()
    logger.info(f'starting {name}')
    yield
    logger.info(f'done {name} {time.time() - start:.3f}s')
    

    
    
def load_entries(filename, ofs, header, selection):
    '''
       **Purpose**
           load entries from pbf file(filename) and stores them in data dictionary
    '''
    data = {Node: [], Way: [], Relation: []}
    with open(filename, 'rb') as fileobj:
        entries = osmformat_pb2.PrimitiveBlock()
        entries.ParseFromString(read_blob(fileobj, ofs, header))
        for entry in read_entries(entries):
            if entry.id not in selection[type(entry)]:
                continue
            data[type(entry)].append(entry)
    return data

def read_entries(block):
    strmap = decode_strmap(block)
    for group in block.primitivegroup:
        for id, tags, lonlat in iter_nodes(block, strmap, group):
            yield Node(id, tags, lonlat)

        for id, refs, tags in iter_ways(block, strmap, group):
            yield Way(id, tags, refs)

        for id, members, tags in iter_relations(block, strmap, group):
            yield Relation(id, tags, members)


def iter_nodes(block, strmap, group):
    dense = group.dense
    if not dense:
        raise ValueError('Only dense nodes are supported')
    granularity = block.granularity or 100
    lat_offset = block.lat_offset or 0
    lon_offset = block.lon_offset or 0
    coord_scale = 0.000000001
    id = lat = lon = tag_pos = 0
    for did, dlat, dlon, tags in zip(
            dense.id, dense.lat, dense.lon,
            parse_tags(strmap, dense.keys_vals)):
        id += did
        lat += coord_scale * (lat_offset + granularity * dlat)
        lon += coord_scale * (lon_offset + granularity * dlon)
        yield (id, tags, (lon, lat))


def iter_ways(block, strmap, group):
    for way in group.ways:
        tags = {
            strmap[k]: strmap[v]
            for k, v in zip(way.keys, way.vals)
        }
        refs = tuple(accumulate(way.refs))
        yield way.id, refs, tags


def iter_relations(block, strmap, group):
    namemap = {}
    for relation in group.relations:
        tags = {
            strmap[k]: strmap[v]
            for k, v in zip(relation.keys, relation.vals)
        }
        refs = tuple(accumulate(relation.memids))
        members = [
            (
                ref,
                namemap.setdefault(
                    rel_type, osmformat_pb2.Relation.MemberType.Name(rel_type)),
                strmap[sid]
            )
            for ref, rel_type, sid in zip(
                    refs, relation.types, relation.roles_sid)
        ]

        yield relation.id, members, tags


def by_id(entry, idset):
    '''
    **Purpose**
        checks if entry id is in the list idset
    '''
    return entry.id in idset

def completefilter(entry,pre_filter):
    '''
    **Purpose**
        checks if an OSM-entry passes the pre_filter
    '''
    filtermap = pre_filter[type(entry)]
    for key in filtermap.keys():
        if key in entry.tags.keys():
            if any(value==True or value in entry.tags.get(key)  for value in filtermap.get(key)):
                return True

def nofilter(entry,way_relation_members):
    '''
    **Purpose**
        checks if an entry is a way and if its id is already stored in the list 
        way_relation_members
    '''
    return type(entry) is Way and entry.id in way_relation_members


# def filter_file_block(filename, ofs, header, filter_func, args, kwargs):
#     with open(filename, 'rb') as file:
#         entries = osmformat_pb2.PrimitiveBlock()
#         entries.ParseFromString(read_blob(file, ofs, header))
#         return [
#             entry
#             for entry in read_entries(entries)
#             if filter_func(entry, *args, **kwargs)
#         ]

def filter_file_block(filename, ofs, header_size, filter_func, args, kwargs):
    with open(filename, 'rb') as file:
        entries = osmformat_pb2.PrimitiveBlock()
        # Pass header_size (an integer) to read_blob
        entries.ParseFromString(read_blob(file, ofs, header_size))
        return [
            entry
            for entry in read_entries(entries)
            if filter_func(entry, *args, **kwargs)
        ]


# def pool_file_query(pool, file):
#     if type(file) is str:
#         file = open(file, 'rb')

#     blocks = [(file.name, ofs, header) for ofs, header in iter_blocks(file)]

#     def query(query_func, *args, **kwargs):
#         entry_lists = pool.starmap(filter_file_block, [
#             block + (query_func, args, kwargs) for block in blocks
#         ])
#         return itertools.chain(*(entries for entries in entry_lists))

#     return query

def pool_file_query(pool, file):
    if type(file) is str:
        file = open(file, 'rb')

    # Ensure header provides the correct size (or equivalent)
    blocks = [(file.name, ofs, header.datasize) for ofs, header in iter_blocks(file)]

    def query(query_func, *args, **kwargs):
        entry_lists = starmap(filter_file_block, [
            (filename, ofs, header_size, query_func, args, kwargs)
            for filename, ofs, header_size in blocks
        ])
        return itertools.chain(*(entries for entries in entry_lists))

    return query
    
    
    
# def pool_file_query2(file):
#     if type(file) is str:
#         file = open(file, 'rb')

#     blocks = [(file.name, ofs, header) for ofs, header in iter_blocks(file)]

#     def query(query_func, *args, **kwargs):
#         entry_lists = starmap(filter_file_block, [
#             block + (query_func, args, kwargs) for block in blocks
#         ])
#         return itertools.chain(*(entries for entries in entry_lists))

#     return query

def pool_file_query2(file):
    if type(file) is str:
        file = open(file, 'rb')

    # Ensure header provides the correct size (or equivalent)
    blocks = [(file.name, ofs, header.datasize) for ofs, header in iter_blocks(file)]

    def query(query_func, *args, **kwargs):
        entry_lists = starmap(filter_file_block, [
            (filename, ofs, header_size, query_func, args, kwargs)
            for filename, ofs, header_size in blocks
        ])
        return itertools.chain(*(entries for entries in entry_lists))

    return query



def filter_pbf(filename, targetname,pre_filter,multiprocess=True):
    """
    **Purpose**
        Parallized pre-Filtering of OSM file by a pre_filter
    **Input** 
        filename:
            PBF file
        pre_filter:
            (see doc: filter)
    **Output**
        targetname:
            JSON-file
    """
    if Path(filename).is_file():
        logger.info(CC.Caption+'PreFilter OSM GAS DATA'+CC.End)
        logger.info('InputFile     : ' + CC.Cyan + filename + CC.End ) 
        filesize=str(os.stat(filename).st_size)
        logger.info('Size          : ' '{:<15}'.format(str(round(int(filesize)/1000))) + ' kbyte')
        timeestimate = float(filesize)/(1750000*4)
        logger.info("Estimated Time: " "{:<15.2f}".format(timeestimate) +' s')
        logger.info("=============================")
    else:
        logger.info(CC.Red + 'OSM_raw_data does not exist' + CC.END)
        sys.exit()
    
    time1 = time.perf_counter()
    if multiprocess==False:
        
        query = pool_file_query2(filename)
        logger.info('0.5')
        entries = list(query(completefilter,pre_filter))
        node_relation_members = set()
        way_relation_members = set()
        way_refs = set()
        relation_way_node_members=set()
        logger.info('1')
        for entry in entries:
            if type(entry) is Relation:
                logger.info('1.1')
                for id, typename, role in entry.members:
                    if typename == 'NODE':
                        logger.info('1.1.1')
                        node_relation_members.add(id)
                    elif typename == 'WAY':
                        logger.info('1.1.2')
                        way_relation_members.add(id)
            if type(entry) is Way:
                way_refs.update(entry.refs)
                logger.info('1.2')
        
            
            
        entries2 = list(query(nofilter, way_relation_members))
        logger.info('1.3')
        logger.info('2')
        for entry in entries2:
                relation_way_node_members.update(entry.refs)
                
        entries.extend(query(by_id, node_relation_members | way_refs | 
                way_relation_members|relation_way_node_members))
        logger.info('3')
        entries.sort(key=lambda entry: entry.id)
        jsondata = {}
        jsondata['Node'] = {
            entry.id: dict(entry._asdict())
            for entry in entries if type(entry) is Node
                
        }
        jsondata['Way'] = {
            entry.id: dict(entry._asdict())
            for entry in entries if type(entry) is Way
        }
        jsondata['Relation'] = {
            entry.id: dict(entry._asdict())
            for entry in entries if type(entry) is Relation
        }

        time2 = time.perf_counter()
        os.makedirs(os.path.dirname(targetname), exist_ok=True)
        #create outputdirectory if it does not exist
        with open(targetname, 'w',encoding="utf-8") as target:
            json.dump(jsondata, target, ensure_ascii=False, indent=4, separators=(',', ':'))

        mesA='Outputfile    : '
        filesize=str(os.stat(targetname).st_size)
        mesB=CC.Cyan + targetname + CC.End +'\n'
        mesC='Size          : ' + '{:<15}'.format(str(round(int(filesize)/1000))) + ' kbyte \n'
        mesD="Time Elapsed  : " "{:<15.2f}".format(time2 - time1) + " s"  + '\n'
        message=mesA + mesB + mesC + mesD + '\n'
        logger.info(message)
    
    
    
    if multiprocess==True:
        with multiprocessing.Pool() as pool:
    #        query = iter_primitive_block( filename)  
            query = pool_file_query(pool, filename)
            logger.info('0.5')
            entries = list(query(completefilter,pre_filter))
            node_relation_members = set()
            way_relation_members = set()
            way_refs = set()
            relation_way_node_members=set()
            logger.info('1')
            for entry in entries:
                if type(entry) is Relation:
                    logger.info('1.1')
                    for id, typename, role in entry.members:
                        if typename == 'NODE':
                            logger.info('1.1.1')
                            node_relation_members.add(id)
                        elif typename == 'WAY':
                            logger.info('1.1.2')
                            way_relation_members.add(id)
                if type(entry) is Way:
                    way_refs.update(entry.refs)
                    logger.info('1.2')
            
                
                
            entries2 = list(query(nofilter, way_relation_members))
            logger.info('1.3')
            logger.info('2')
            for entry in entries2:
                    relation_way_node_members.update(entry.refs)
                    
            entries.extend(query(by_id, node_relation_members | way_refs | 
                    way_relation_members|relation_way_node_members))
            logger.info('3')
            entries.sort(key=lambda entry: entry.id)
            jsondata = {}
            jsondata['Node'] = {
                entry.id: dict(entry._asdict())
                for entry in entries if type(entry) is Node
                    
            }
            jsondata['Way'] = {
                entry.id: dict(entry._asdict())
                for entry in entries if type(entry) is Way
            }
            jsondata['Relation'] = {
                entry.id: dict(entry._asdict())
                for entry in entries if type(entry) is Relation
            }

            time2 = time.perf_counter()
            os.makedirs(os.path.dirname(targetname), exist_ok=True)
            #create outputdirectory if it does not exist
            with open(targetname, 'w',encoding="utf-8") as target:
                json.dump(jsondata, target, ensure_ascii=False, indent=4, separators=(',', ':'))

            mesA='Outputfile    : '
            filesize=str(os.stat(targetname).st_size)
            mesB=CC.Cyan + targetname + CC.End +'\n'
            mesC='Size          : ' + '{:<15}'.format(str(round(int(filesize)/1000))) + ' kbyte \n'
            mesD="Time Elapsed  : " "{:<15.2f}".format(time2 - time1) + " s"  + '\n'
            message=mesA + mesB + mesC + mesD + '\n'
            logger.info(message)
