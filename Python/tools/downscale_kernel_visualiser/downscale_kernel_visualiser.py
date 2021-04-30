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

import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from asemi_segmenter.lib import downscales
from asemi_segmenter.lib import regions
from asemi_segmenter.lib import volumes

#########################################
def visualise(in_array, sigma):
    downsample_kernel = downscales.GaussianDownsampleKernel(sigma=sigma)
    out_array1 = downscales.downscale(in_array, downsample_kernel, 1, remove_pad=False)
    out_array2 = downscales.downscale(in_array, downsample_kernel, 2, remove_pad=False)

    (fig, ax) = plt.subplots(1, 1)
    kernel = downsample_kernel.get_kernel(1)
    ax.set_title('Sum of kernel values: {}'.format(kernel.sum()))
    ax.matshow(kernel[kernel.shape[0]//2,:,:], cmap='gray')
    fig.show()

    (fig, axs) = plt.subplots(3, 3, figsize=(12,8))

    min_value = np.min(in_array)
    max_value = np.max(in_array)
    for (row, array) in enumerate([ in_array, out_array1, out_array2 ]):
        for col in range(3):
            img = np.rot90([
                    array[array.shape[0]//2,:,:],
                    array[:,array.shape[1]//2,:],
                    array[:,:,array.shape[2]//2],
                ][col])
            img = regions.get_neighbourhood_array_2d(img, [img.shape[0]//2, img.shape[1]//2], 50, {0,1,2})
            axs[row,col].matshow(img, cmap='plasma', vmin=min_value, vmax=max_value)
            axs[row,col].set_xlabel([ 'row', 'slice', 'row' ][col])
            axs[row,col].set_ylabel([ 'column', 'column', 'slice' ][col])
            axs[row,col].set_xticks([], [])
            axs[row,col].set_yticks([], [])

    fig.tight_layout()
    fig.show()

#########################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume_fullfname', required=True,
        help='Full file name (with path) of volume file to downscale.')
    parser.add_argument('--sigma', required=True, type=float,
        help='Sigma of Gaussian kernel to use when downscaling.')
    args = parser.parse_args()

    print('Running...')

    full_volume = volumes.FullVolume(args.volume_fullfname)
    full_volume.load(as_readonly=True)
    in_array = full_volume.get_scale_array(0)[:]

    visualise(in_array, sigma=args.sigma)

    input('Press enter to exit.')


# main entry point
if __name__ == '__main__':
   main()
