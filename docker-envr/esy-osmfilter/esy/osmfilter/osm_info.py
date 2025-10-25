#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module analysis the data dictionary for tags
"""

import json, os, sys
import logging

logging.basicConfig()
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def data2NodeWayRelation(data):
    """
    **Purpose**
        Splits data dictionary into 3 dictionaries for nodes, ways and relations
    """
    return data["Node"],data["Way"],data["Relation"]

class Tee(object):
    """
    **Purpose**
        Parallel output to stdout and logfile=f
    **Example**
        original = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        sys.stdout = original
    """
    def __init__(self,*files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

def showtagcount(OSMdict):
    """
    **Purpose**
        Calculates the tagcount for each tag in OSMdict
    **Input**
        OSMdict
            prefiltered Data
    **Output**
        Writes list with tags and their occurence count to stdout
    """
    sys.stdout.write("============\n") 
    taglist={}
    totalcount=0
    for i in range(len(OSMdict)):
        if "tags" in list(OSMdict[i].keys()):
            for keyitem in list(OSMdict[i]["tags"].keys()):
                if keyitem not in taglist:
                    taglist.update({keyitem:1})
                    totalcount+=1
                else:
                    taglist.update({keyitem:int(taglist[keyitem])+1})
                    totalcount+=1
                    
    taglistsort=sorted(taglist,key=taglist.__getitem__,reverse=True)
    sys.stdout.write("{0:40s} {1:>8s}".format("total:", str(totalcount)) +"\n")
    sys.stdout.write("============\n")
    for key in taglistsort:
        sys.stdout.write("{0:40s} {1:>8s}".format(key,str(taglist[key]))+"\n")
pass

def ReadJason(inputfile,verbose="yes"):
    """
    **Purpose**
        Reading prefiltered OSM data from JSON file into dictionary
        Writes Nodes,Ways,Relation count into logfile
    **Input**
        inputfile:
            dirpathname of JSON-file
    **Calls** 
        showtagcount
    **Output** 
        Data:
           dictionary of prefiltered data
    """

    with open(inputfile,encoding="utf-8") as f:
        data = json.load(f)
    #logdirbase=(os.path.dirname(inputfile)+"/Analysis/")
    
    logdirbase=(os.path.dirname(inputfile))
    logfile=os.path.join(logdirbase,"taglist.log")
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    f = open(logfile, 'w',encoding="utf-8")
       # f = open('/home/apluta/SciGrid/SciGRID_gas/Ausgabe/OSM/Brandenburg/taglist.txt', 'w')
    original = sys.stdout
    if verbose=="yes":
        sys.stdout = Tee(sys.stdout, f)
    else:
        out = open(os.devnull, 'w',encoding="utf-8")
        sys.stdout = Tee(f)
        #sys.stdout=os.devnull

    sys.stdout.write("Overview OSM data" +"\n")
    sys.stdout.write("===========\n")
    sys.stdout.write("Nodes:       " + "{:>26}".format(str(len(data["Node"])))+"\n")
    sys.stdout.write("Ways:        " + "{:>26}".format(str(len(data["Way"])))+"\n")
    sys.stdout.write("Relations:   " + "{:>26}".format(str(len(data["Relation"])))+"\n")
    sys.stdout.write("===========\n\n\n")
    
    nodes=[]
    ways=[]
    relations=[]

    for entry in data["Relation"]:
        relations.append(data["Relation"][entry])
    sys.stdout.write("\n\n")  
    sys.stdout.write("Tag Count Relations:\n")

    showtagcount(relations)
    
    for entry in data["Way"]:
         ways.append(data["Way"][entry])
    sys.stdout.write("\n\n")  
    sys.stdout.write("Tag Count Ways:\n")

    showtagcount(ways)
    
    for entry in data["Node"]:
         nodes.append(data["Node"][entry])
    sys.stdout.write("\n\n")  
    sys.stdout.write("Tag Count Nodes:\n")
    
    showtagcount(nodes)
    sys.stdout = original

    return data



