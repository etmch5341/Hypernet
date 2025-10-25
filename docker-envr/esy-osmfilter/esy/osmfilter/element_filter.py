#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to applicate the whitelist and blacklist filter for elements
"""

# def filter4entry(data,blackfilter,whitefilter_entry):
#     """
#     **Purpose**
#         filter prefiltered Data for elements 
#     **Input** 
#         Data:
#             dictionary of prefiltered data
#         blackfilter:
#             blackfilterlist (see doc: filter) 
#             [filter](filter.md)
#         entry:
#             entry from whitefilter (see doc: filter)
#     **Output**
#         filterIDlist
#             list of IDs which pass the check
#     """
#     filterIDlist=[]
#     itemtypes=["Way","Relation","Node"]

#     for itemtype in itemtypes:
#         for entry in data[itemtype]:
#             if all(item in data[itemtype][entry]["tags"].items() for item in whitefilter_entry):
#                 if blackfilter!='':                
#                     if not any(item in list(data[itemtype][entry]["tags"].items()) for item in blackfilter):
#                         filterIDlist.append((str(data[itemtype][entry]["id"]),itemtype))
#                 else:
#                     filterIDlist.append((str(data[itemtype][entry]["id"]),itemtype))
#     return filterIDlist

# Using "any()" instead of "all()" for partial matching
def filter4entry(data, blackfilter, whitefilter_entry):
    filterIDlist = []
    itemtypes = ["Way", "Relation", "Node"]

    for itemtype in itemtypes:
        for entry in data[itemtype]:
            entry_tags = data[itemtype][entry]["tags"]
            print(f"Checking {itemtype} {entry} with tags: {entry_tags}")  # Debugging output
            if any(item in entry_tags.items() for item in whitefilter_entry):
                if blackfilter != '':
                    if not any(item in list(entry_tags.items()) for item in blackfilter):
                        filterIDlist.append((str(data[itemtype][entry]["id"]), itemtype))
                else:
                    filterIDlist.append((str(data[itemtype][entry]["id"]), itemtype))
    return filterIDlist



def sum_filter_results(*args):
    """
    **Purpose**
        helper method to sums up all filter results
    **Return**
        returning unique list of OSM ids, that fulfill filter request
    """
    all_results = []
    all_results = args[0][:]
    for i in range(len(args)):
        if i!=0:
            all_results.extend(args[i][:])
    return list(set(all_results))

def GetOSM_IDs(data,whitefilter,blackfilter):
    """
    **Purpose**
        method to sums up all filter results
    **Return**
        returning unique list of OSM ids, that fulfill filter request
    """
    filter_results=[]
    for entry in whitefilter:
        filter_result=filter4entry(data,blackfilter,entry)
        filter_results.append(filter_result)
    result=sum_filter_results(*filter_results)
    return result
