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

'''
Module for collecting subarrays (regions) from larger arrays.

Given a 1D array, a subarray can be collected by either giving the range of values to extract or by
giving a center index and a radius from the center to extract (called a neighbourhood). Here is an
example:

Array:    abcdefghi
Range:    slice(2, 6)
Subarray: cdef

Array:    abcdefghi
Center:   (4,)
Radius:   2
Subarray: cdefg

A subarray can also lie outside of the array, in which case pad values will fill in the missing
values. Here is an example:

Array:    abcdefghi
Range:    slice(6, 11)
Subarray: ghi00

Array:    abcdefghi
Range:    slice(-2, 3)
Subarray: 00abc

A scale can also be provided in order to be able to specify ranges and centers according to a full
sized array but adapt them to the corresponding location in smaller array (such that the subarray
is extracted from the smaller array).

These features are also available for 2D and 3D arrays. In this case, when extracting a
neighbourhood, you also specify the neighbouring_dims parameter which is a set specifying which
dimensions from the neighbourhood square/cube to keep. Specifying a neighbouring_dims of {0,1,2}
in a 3D array's neighbourhood will keep the whole cube but {1,2} will only return a square
consisting of the second and third dimensions (a slice) with the original central voxel still being
at the center. Likewise neighbouring_dims can be just {} which means that only the central voxel
is to be returned.
'''

import os
import numpy as np
import sys
from asemi_segmenter.lib import downscales

#########################################
def get_subarray_1d(array_1d, col_slice, pad=0, scale=0):
    '''
    Get a subarray from a 1D array.

    :param numpy.ndarray array_1d: The array.
    :param slice col_slice: Python slice of indexes to extract.
    :param int pad: The pad value.
    :param int scale: The scale to adapt the position to.
    :return: The subarray.
    :rtype: numpy.ndarray
    '''
    if col_slice.step is not None:
        raise ValueError()

    [ num_cols ] = array_1d.shape

    col_slice = downscales.downscale_slice(col_slice, scale)
    col_slice = slice(
            col_slice.start if col_slice.start is not None else 0,
            col_slice.stop  if col_slice.stop  is not None else num_cols
        )
    col_shift = max(0, 0-col_slice.start)
    available_col_slice = slice(max(0, col_slice.start), min(num_cols, col_slice.stop))

    full_subregion_shape = [
            col_slice.stop - col_slice.start
        ]
    available_subregion_shape = [
            available_col_slice.stop - available_col_slice.start
        ]

    if available_subregion_shape[0] <= 0:
        subregion = np.full(full_subregion_shape, pad, array_1d.dtype)
    else:
        subregion = array_1d[available_col_slice]
        if (
            col_slice.start < 0 or col_slice.stop > num_cols
        ):
            full_subregion = np.full(full_subregion_shape, pad, array_1d.dtype)
            full_subregion[
                    col_shift:col_shift+available_subregion_shape[0]
                ] = subregion

            subregion = full_subregion

    return subregion

#########################################
def get_neighbourhood_array_1d(array_1d, center, radius, pad=0, scale=0, scale_radius=False):
    '''
    Get a neighbourhood from a 1D array.

    :param numpy.ndarray array_1d: The array.
    :param tuple center: Tuple with the coordinates of the neighbourhood center.
    :param int radius: The neighbourhood radius.
    :param int pad: The pad value.
    :param int scale: The scale to adapt the position to.
    :parama bool scale_radius: Whether to also change the radius to the corresponding scale or
        leave it the same (and cover a larger area of the original volume).
    :return: The subarray.
    :rtype: numpy.ndarray
    '''
    if scale_radius:
        radius //= 2**scale
    subvolume_slices = [
            slice(c-radius, c+radius+1)
            for (i, c) in enumerate(downscales.downscale_pos(center, scale))
        ]
    return get_subarray_1d(array_1d, *subvolume_slices, pad)

#########################################
def get_subarray_2d(array_2d, row_slice, col_slice, to_1d=False, pad=0, scale=0):
    '''
    Get a subarray from a 2D array.

    :param numpy.ndarray array_2d: The array.
    :param slice row_slice: Python slice of indexes to extract in terms of rows.
    :param slice col_slice: Python slice of indexes to extract in terms of columns.
    :param bool to_1d: Whether to convert the resulting array to 1D if the subarray shape has
        exactly one non-1 dimension.
    :param int pad: The pad value.
    :param int scale: The scale to adapt the position to.
    :return: The subarray.
    :rtype: numpy.ndarray
    '''
    if row_slice.step is not None:
        raise ValueError()
    if col_slice.step is not None:
        raise ValueError()

    [ num_rows, num_cols ] = array_2d.shape

    row_slice = downscales.downscale_slice(row_slice, scale)
    row_slice = slice(
            row_slice.start if row_slice.start is not None else 0,
            row_slice.stop  if row_slice.stop  is not None else num_rows
        )
    row_shift = max(0, 0-row_slice.start)
    available_row_slice = slice(max(0, row_slice.start), min(num_rows, row_slice.stop))

    col_slice = downscales.downscale_slice(col_slice, scale)
    col_slice = slice(
            col_slice.start if col_slice.start is not None else 0,
            col_slice.stop  if col_slice.stop  is not None else num_cols
        )
    col_shift = max(0, 0-col_slice.start)
    available_col_slice = slice(max(0, col_slice.start), min(num_cols, col_slice.stop))

    flat_dims = []
    if to_1d:
        for (i, s) in enumerate([ row_slice, col_slice ]):
            if s.stop - s.start == 1:
                flat_dims.append(i)
        if len(flat_dims) != 1:
            raise ValueError()

    full_subregion_shape = [
            row_slice.stop - row_slice.start,
            col_slice.stop - col_slice.start
        ]
    available_subregion_shape = [
            available_row_slice.stop - available_row_slice.start,
            available_col_slice.stop - available_col_slice.start
        ]

    if available_subregion_shape[0] <= 0 or available_subregion_shape[1] <= 0:
        subregion = np.full(full_subregion_shape, pad, array_2d.dtype)
    else:
        subregion = array_2d[available_row_slice, available_col_slice]
        if (
            row_slice.start < 0 or row_slice.stop > num_rows or
            col_slice.start < 0 or col_slice.stop > num_cols
        ):
            full_subregion = np.full(full_subregion_shape, pad, array_2d.dtype)
            full_subregion[
                    row_shift:row_shift+available_subregion_shape[0],
                    col_shift:col_shift+available_subregion_shape[1]
                ] = subregion

            subregion = full_subregion

    if flat_dims != []:
        new_slices = [
                slice(subregion.shape[0]),
                slice(subregion.shape[1]),
            ]
        for i in flat_dims:
            new_slices[i] = 0
        new_slices = tuple(new_slices)
        subregion = subregion[new_slices]

    return subregion

#########################################
def get_neighbourhood_array_2d(array_2d, center, radius, neighbouring_dims, pad=0, scale=0, scale_radius=False):
    '''
    Get a neighbourhood from a 2D array.

    :param numpy.ndarray array_2d: The array.
    :param tuple center: Tuple with the coordinates of the neighbourhood center.
    :param int radius: The neighbourhood radius.
    :param set neighbouring_dims: The dimensions of the neighbourhood to keep.
    :param int pad: The pad value.
    :param int scale: The scale to adapt the position to.
    :parama bool scale_radius: Whether to also change the radius to the corresponding scale or
        leave it the same (and cover a larger area of the original volume).
    :return: The subarray.
    :rtype: numpy.ndarray
    '''
    if scale_radius:
        radius //= 2**scale
    subvolume_slices = [
            slice(c-radius, c+radius+1) if i in neighbouring_dims else slice(c, c+1)
            for (i, c) in enumerate(downscales.downscale_pos(center, scale))
        ]
    return get_subarray_2d(array_2d, *subvolume_slices, len(neighbouring_dims) == 1, pad)

#########################################
def get_subarray_3d(array_3d, slice_slice, row_slice, col_slice, to_2d=False, to_1d=False, pad=0, scale=0):
    '''
    Get a subarray from a 3D array.

    :param numpy.ndarray array_3d: The array.
    :param slice slice_slice: Python slice of indexes to extract in terms of slices (depth).
    :param slice row_slice: Python slice of indexes to extract in terms of rows.
    :param slice col_slice: Python slice of indexes to extract in terms of columns.
    :param bool to_2d: Whether to convert the resulting array to 2D if the subarray shape has
        exactly one non-1 dimension.
    :param bool to_1d: Whether to convert the resulting array to 1D if the subarray shape has
        exactly two non-1 dimensions.
    :param int pad: The pad value.
    :param int scale: The scale to adapt the position to.
    :return: The subarray.
    :rtype: numpy.ndarray
    '''
    if slice_slice.step is not None:
        raise ValueError()
    if row_slice.step is not None:
        raise ValueError()
    if col_slice.step is not None:
        raise ValueError()
    if to_2d and to_1d:
        raise ValueError()

    [ num_slices, num_rows, num_cols ] = array_3d.shape

    slice_slice = downscales.downscale_slice(slice_slice, scale)
    slice_slice = slice(
            slice_slice.start if slice_slice.start is not None else 0,
            slice_slice.stop  if slice_slice.stop  is not None else num_slices
        )
    slice_shift = max(0, 0-slice_slice.start)
    available_slice_slice = slice(max(0, slice_slice.start), min(num_slices, slice_slice.stop))

    row_slice = downscales.downscale_slice(row_slice, scale)
    row_slice = slice(
            row_slice.start if row_slice.start is not None else 0,
            row_slice.stop  if row_slice.stop  is not None else num_rows
        )
    row_shift = max(0, 0-row_slice.start)
    available_row_slice = slice(max(0, row_slice.start), min(num_rows, row_slice.stop))

    col_slice = downscales.downscale_slice(col_slice, scale)
    col_slice = slice(
            col_slice.start if col_slice.start is not None else 0,
            col_slice.stop  if col_slice.stop  is not None else num_cols
        )
    col_shift = max(0, 0-col_slice.start)
    available_col_slice = slice(max(0, col_slice.start), min(num_cols, col_slice.stop))

    flat_dims = []
    if to_2d or to_1d:
        for (i, s) in enumerate([ slice_slice, row_slice, col_slice ]):
            if s.stop - s.start == 1:
                flat_dims.append(i)
        if to_2d and len(flat_dims) != 1:
            raise ValueError()
        elif to_1d and len(flat_dims) != 2:
            raise ValueError()

    full_subregion_shape = [
            slice_slice.stop - slice_slice.start,
            row_slice.stop - row_slice.start,
            col_slice.stop - col_slice.start
        ]
    available_subregion_shape = [
            available_slice_slice.stop - available_slice_slice.start,
            available_row_slice.stop - available_row_slice.start,
            available_col_slice.stop - available_col_slice.start
        ]
    if available_subregion_shape[0] <= 0 or available_subregion_shape[1] <= 0 or available_subregion_shape[2] <= 0:
        subregion = np.full(full_subregion_shape, pad, array_3d.dtype)
    else:
        subregion = array_3d[available_slice_slice, available_row_slice, available_col_slice]
        if (
            slice_slice.start < 0 or slice_slice.stop > num_slices or
            row_slice.start < 0 or row_slice.stop > num_rows or
            col_slice.start < 0 or col_slice.stop > num_cols
        ):
            full_subregion = np.full([
                    slice_slice.stop - slice_slice.start,
                    row_slice.stop - row_slice.start,
                    col_slice.stop - col_slice.start
                ], pad, array_3d.dtype)
            full_subregion[
                    slice_shift:slice_shift+available_subregion_shape[0],
                    row_shift:row_shift+available_subregion_shape[1],
                    col_shift:col_shift+available_subregion_shape[2]
                ] = subregion

            subregion = full_subregion

    if len(flat_dims) > 0:
        new_slices = [
                slice(subregion.shape[0]),
                slice(subregion.shape[1]),
                slice(subregion.shape[2])
            ]
        for i in flat_dims:
            new_slices[i] = 0
        new_slices = tuple(new_slices)
        subregion = subregion[new_slices]

    return subregion

#########################################
def get_neighbourhood_array_3d(array_3d, center, radius, neighbouring_dims, pad=0, scale=0, scale_radius=False):
    '''
    Get a neighbourhood from a 3D array.

    :param numpy.ndarray array_3d: The array.
    :param tuple center: Tuple with the coordinates of the neighbourhood center.
    :param int radius: The neighbourhood radius.
    :param set neighbouring_dims: The dimensions of the neighbourhood to keep.
    :param int pad: The pad value.
    :param int scale: The scale to adapt the position to.
    :parama bool scale_radius: Whether to also change the radius to the corresponding scale or
        leave it the same (and cover a larger area of the original volume).
    :return: The subarray.
    :rtype: numpy.ndarray
    '''
    if scale_radius:
        radius //= 2**scale
    subvolume_slices = [
            slice(c-radius, c+radius+1) if i in neighbouring_dims else slice(c, c+1)
            for (i, c) in enumerate(downscales.downscale_pos(center, scale))
        ]
    return get_subarray_3d(array_3d, *subvolume_slices, len(neighbouring_dims) == 2, len(neighbouring_dims) == 1, pad)
