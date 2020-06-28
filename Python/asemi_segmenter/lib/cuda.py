'''Module for CUDA code.'''

import numpy as np
import math
import pkg_resources
from string import Template

import pycuda
try:
    import pycuda.autoinit
    import pycuda.driver as drv
    import pycuda.compiler
    gpu_available = True
except pycuda._driver.Error:
    gpu_available = False

#########################################
class histograms():
    """Histogram functions"""
    __initialised = False
    def __init__(self):
        if gpu_available and not self.__class__.__initialised:
            # load and compile CUDA code
            code = pkg_resources.resource_string('asemi_segmenter.resources.cuda', 'histograms.cu').decode()
            code = Template(code)
            code = code.substitute(data_t='uint16_t', result_t='float', index_t='uint8_t')
            self.__class__.__mod = pycuda.compiler.SourceModule(code)
            # get function pointers
            self.__class__.histogram_3d = self.__class__.__mod.get_function("histogram_3d")
            self.__class__.histogram_2d_xy = self.__class__.__mod.get_function("histogram_2d_xy")
            self.__class__.histogram_2d_xz = self.__class__.__mod.get_function("histogram_2d_xz")
            self.__class__.histogram_2d_yz = self.__class__.__mod.get_function("histogram_2d_yz")
            # mark as done
            self.__class__.__initialised = True
