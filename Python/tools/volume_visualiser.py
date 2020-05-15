#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from asemi_segmenter.lib import regions
from asemi_segmenter.lib import volumes

matplotlib.use('TkAgg')

#########################################
def _expand_slice(s):
    return slice(s.start, s.stop + 1)

#########################################
def display_volume(in_array, relative_intensities=True, cross_section=None):
    shape = in_array.shape

    mask = np.full(shape, True, np.bool)
    if len(shape) == 3 and cross_section is not None:
        if cross_section == 'diagonal':
            for z in range(shape[2]):
                for x in range(int(z/shape[2]*shape[0])):
                    for y in range(int(z/shape[2]*shape[1])):
                        mask[shape[0]-1-x, y, z] = False
        elif cross_section == 'corner':
            mask[shape[0]//2:, :shape[1]//2, shape[2]//2:] = False

    max_value = in_array.max()
    min_value = in_array.min()
    if not relative_intensities:
        dtype_info = np.iinfo(in_array)
        max_value = dtype_info.max
        min_value = dtype_info.min
    corrected_values = (in_array - min_value)/(max_value - min_value + 1e-200)

    cmap = plt.cm.get_cmap('plasma')
    colours = cmap(corrected_values, 1.0)

    if len(shape) == 3:
        print('This may take a while...')
        fig = plt.figure(figsize=(12,8))

        ax = fig.add_subplot(2, 2, 1, projection='3d')
        indices = np.mgrid[0:shape[0]+1, 0:shape[1]+1, 0:shape[2]+1]
        ax.voxels(indices[0], indices[1], indices[2], mask, facecolors=colours)
        ax.set_xlabel('slice')
        ax.set_ylabel('row')
        ax.set_zlabel('column')
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.set_zticks([], [])

        ax = fig.add_subplot(2, 2, 2)
        ax.imshow(np.rot90(colours[shape[0]//2,:,:]))
        ax.set_xlabel('row')
        ax.set_ylabel('column')
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        ax = fig.add_subplot(2, 2, 3)
        ax.imshow(np.rot90(colours[:,shape[1]//2,:]))
        ax.set_xlabel('slice')
        ax.set_ylabel('column')
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        ax = fig.add_subplot(2, 2, 4)
        ax.imshow(np.rot90(colours[:,:,shape[2]//2]))
        ax.set_xlabel('row')
        ax.set_ylabel('slice')
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        fig.tight_layout()
        fig.show()
    else:
        (fig, ax) = plt.subplots(1, 1, figsize=(12,8))
        ax.imshow(colours)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        fig.tight_layout()
        fig.show()

#########################################

full_volume = volumes.FullVolume(os.path.join('..', 'example_volume', 'output', 'preprocess', 'volume.hdf'))
full_volume.load()
in_array = full_volume.get_scale_array(1)[:]

display_volume(in_array, relative_intensities=True, cross_section='corner')
input('Press enter to exit.')
