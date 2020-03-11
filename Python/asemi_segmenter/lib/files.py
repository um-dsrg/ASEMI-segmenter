import os, errno
import shutil

#########################################
def mkdir(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

#########################################
def delete(path):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except FileNotFoundError as e:
        pass

#########################################
def fexists(path):
    return os.path.isdir(path) or os.path.isfile(path)
