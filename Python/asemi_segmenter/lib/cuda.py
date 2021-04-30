#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti
#
# This file is part of ASEMI-segmenter.
#
# ASEMI-segmenter is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ASEMI-segmenter is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASEMI-segmenter.  If not, see <http://www.gnu.org/licenses/>.

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
