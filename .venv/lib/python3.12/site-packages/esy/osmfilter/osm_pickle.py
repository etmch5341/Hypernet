# -*- coding: utf-8 -*-
"""
Module for pickle load and saving.
"""
import pickle
import os


def picklesave(elements, filedir):
    ''' 
    **Purpose**
        Saves elements dictionary at filedir
    '''
    
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    for elementname  in elements:
        with open(os.path.join(filedir, elementname+'.pickle'), 'wb') as handle:
            pickle.dump(elements[elementname], handle, protocol=pickle.HIGHEST_PROTOCOL)
    return           

def pickleload(elements, filedir):
    ''' 
    **Purpose**
        Restore elements dictionary from filedir
    '''
    for elementname  in elements:
        with open(os.path.join(filedir, elementname+'.pickle'), 'rb') as handle:
            elements[elementname] = pickle.load(handle)
    return elements


#def picklesavelist(element, filename, filedir):
#    ''' 
#    **Purpose**
#        Pickles list to file directory
#    
#    **Input**
#    element
#            dictionary 
#            filedirectory
#    '''
#    if not os.path.exists(filedir):
#        os.makedirs(filedir)
#    with open(os.path.join(filedir,filename), 'wb') as handle:
#        pickle.dump(element, handle, protocol=pickle.HIGHEST_PROTOCOL)
#    return
#
#def pickleloadlist(filedirname,file):
#    ''' UnPickles list from filedirectory
#    '''
#
#    with open(os.path.join(filedirname,file), 'rb') as handle:
#        liste=pickle.load(handle)
#    return liste
