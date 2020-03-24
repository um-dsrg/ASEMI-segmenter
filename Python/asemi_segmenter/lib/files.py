'''Module for file related functions.'''

import os, errno
import shutil

#########################################
def mkdir(dir):
    '''
    Create a directory recursively and ignore if it already exists.
    
    :param str dir: The directory path to create.
    '''
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

#########################################
def delete(path):
    '''
    Delete a file or directory and ignore if it already exists.
    
    :param str path: The path to the object to delete.
    '''
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except FileNotFoundError as e:
        pass

#########################################
def fexists(path):
    '''
    Check if a file or directory exists.
    
    :param str path: The path to the object to check.
    '''
    return os.path.isdir(path) or os.path.isfile(path)
