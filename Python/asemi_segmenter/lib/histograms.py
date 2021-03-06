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

'''Module for histogram related functions.'''

import os
import numpy as np
import fast_histogram
import sys
import math
from asemi_segmenter.lib import regions
from asemi_segmenter.lib import featurisers
from asemi_segmenter.lib import cuda

#########################################
def histogram(array, num_bins, value_range):
    '''
    Compute a histogram from a range and number of bins.

    :param numpy.ndarray array: The array of values.
    :param int num_bins: The number of bins in the histogram.
    :param tuple value_range: A 2-tuple consisting of the minimum and maximum values to consider
        with the maximum not being inclusive.
    :return: A 1D histogram array.
    :rtype: numpy.ndarray
    '''
    return fast_histogram.histogram1d(array, num_bins, value_range)
    # Some times fast_histogram fails test cases because of a minor bug (https://github.com/astrofrog/fast-histogram/issues/45) so this may be useful during testing.
    # return np.histogram(array, num_bins, value_range)[0]

#########################################
def apply_histogram_to_all_neighbourhoods_in_slice_3d(array_3d, slice_range, radius, neighbouring_dims, min_range, max_range, num_bins, pad=0, row_slice=slice(None), col_slice=slice(None)):
    '''
    Find a histograms of the values in the neighbourhood around every element
    in a volume slice.

    Given a slice of voxels in a volume, this function will find the histogram
    of values around every voxel in the slice. The histogram around a voxel is
    computed using a cube of a given radius centered around the voxel. Radius
    defined such that the side of the cube is radius + 1 + radius long.

    Histograms of neighbourhoods around every voxel in a slice can be computed
    efficiently using a rolling algorithm that reuses histograms in other
    neighbouring voxels. Consider if we had to use this function on a 2D
    image instead of a volume and using a 2D square neighbourhood instead
    of a cube. We'll be using this 3x3 image as an example:
        [a,b,c]
        [d,e,f]
        [g,h,i]
    The neighbourhood around the pixel at (0,0) with radius 1 has the
    following frequencies:
        PAD=5, a=1, b=1, c=0, d=1, e=1, f=0, g=0, h=0, i=0 => [5,1,1,0,1,1,0,0,0,0]
    The neighbourhood around (0,1) has values in common with the
    neighbourhood around (0,0) so we can avoid counting all the elements in
    this neighbourhood by instead counting only what has changed from the
    previous neighbourhood and update the frequencies with new information
    in the dropped and new columns:
        histogram_01 = histogram_00 - histogram([PAD,a,d]) + histogram([PAD,c,f])
    Likewise, the neighbourhood around (1,0) has values in common with (0,0)
    as well and can be calculated by subtracting the dropped row and adding
    the new column:
        histogram_10 = histogram_00 - histogram([PAD,PAD,PAD]) + histogram([g,h,i])
    This means that we only need to perform a full histogram count for the
    top left corner as everything else can be computed by rolling the values.

    For extracting histograms in 3D space, we will still be only processing
    a single slice but with adjacent slices for context. Given a 3D array A,
    the neighbourhood around A[r,c] (in the slice of interest) with radius R
    and neighbouring_dims d is hist(A[r,c], R, d). For each d, the following
    optimisations can be performed:
        hist(A[r,c], R, {0,1,2}) = hist(A[r-1,c], R, {0,1,2})
                                    - hist(A[r-1-R,c], R, {0,2})
                                    + hist(A[r+R,c], R, {0,2})
                          (also) = hist(A[r,c-1], R, {0,1,2})
                                    - hist(A[r,c-1-R], R, {0,1})
                                    + hist(A[r,c+R], R, {0,1})

    :param numpy.ndarray array_3d: The volume from which to extract the histograms.
    :param slice slice_range: The range of slices to use within the volume.
    :param int radius: The radius of the neighbourhood around each voxel.
    :param set neighbouring_dims: The set of dimensions to apply the neighbourhoods on.
    :param int min_range: The minimum range of the values to consider.
    :param int max_range: The maximum range of the values to consider, not included.
    :param int num_bins: The number of bins in the histograms.
    :param int pad: The pad value for values outside the array.
    :param slice row_slice: The range of rows in the slice to consider.
    :param slice col_slice: The range of columns in the slice to consider.
    :return: A 3D array where the first two dimensions are equal to the dimensions of the slice
        and the last dimension is the number of bins.
    :rtype: numpy.ndarray
    '''
    [ num_slcs_in, num_rows_in, num_cols_in ] = array_3d.shape
    slc_slice = slice(
        slice_range.start if slice_range.start is not None else 0,
        slice_range.stop  if slice_range.stop is not None else num_slcs_in
        )
    row_slice = slice(
        row_slice.start if row_slice.start is not None else 0,
        row_slice.stop  if row_slice.stop is not None else num_rows_in
        )
    col_slice = slice(
        col_slice.start if col_slice.start is not None else 0,
        col_slice.stop  if col_slice.stop is not None else num_cols_in
        )
    num_slcs_out = slc_slice.stop - slc_slice.start
    num_rows_out = row_slice.stop - row_slice.start
    num_cols_out = col_slice.stop - col_slice.start

    result = np.empty([ num_slcs_out, num_rows_out, num_cols_out, num_bins ], featurisers.feature_dtype)
    if neighbouring_dims == {0,1,2}:
        for (slc_out, slc_in) in enumerate(range(slc_slice.start, slc_slice.stop)):
            for (row_out, row_in) in enumerate(range(row_slice.start, row_slice.stop)):
                for (col_out, col_in) in enumerate(range(col_slice.start, col_slice.stop)):
                    if col_out == 0 and row_out == 0:
                        result[slc_out, row_out, col_out, :] = histogram(regions.get_neighbourhood_array_3d(array_3d, (slc_in, row_in, col_in), radius, {0,1,2}, pad), num_bins, (min_range, max_range)) #Get the only completely computed histogram (top left corner).
                    elif col_out == 0:
                        result[slc_out, row_out, col_out, :] = (
                                result[slc_out, row_out-1, col_out, :]
                                -
                                histogram(regions.get_neighbourhood_array_3d(array_3d, (slc_in, row_in-1-radius, col_in), radius, {0,2}, pad), num_bins, (min_range, max_range)) #Undo effect of dropped row face.
                                +
                                histogram(regions.get_neighbourhood_array_3d(array_3d, (slc_in, row_in+radius, col_in), radius, {0,2}, pad), num_bins, (min_range, max_range)) #Include effect of new row face.
                            )
                    else:
                        result[slc_out, row_out, col_out, :] = (
                                result[slc_out, row_out, col_out-1, :]
                                -
                                histogram(regions.get_neighbourhood_array_3d(array_3d, (slc_in, row_in, col_in-1-radius), radius, {0,1}, pad), num_bins, (min_range, max_range)) #Undo effect of dropped column face.
                                +
                                histogram(regions.get_neighbourhood_array_3d(array_3d, (slc_in, row_in, col_in+radius), radius, {0,1}, pad), num_bins, (min_range, max_range)) #Include effect of new column face.
                            )

    elif neighbouring_dims == {0,1}:
        for (slc_out, slc_in) in enumerate(range(slc_slice.start, slc_slice.stop)):
            for (row_out, row_in) in enumerate(range(row_slice.start, row_slice.stop)):
                for (col_out, col_in) in enumerate(range(col_slice.start, col_slice.stop)):
                    if row_out == 0:
                        result[slc_out, row_out, col_out, :] = histogram(regions.get_neighbourhood_array_3d(array_3d, (slc_in, row_in, col_in), radius, {0,1}, pad), num_bins, (min_range, max_range)) #Get the only completely computed histograms (top row)
                    else:
                        result[slc_out, row_out, col_out, :] = (
                                result[slc_out, row_out-1, col_out, :]
                                -
                                histogram(regions.get_neighbourhood_array_3d(array_3d, (slc_in, row_in-1-radius, col_in), radius, {0}, pad), num_bins, (min_range, max_range)) #Undo effect of dropped row
                                +
                                histogram(regions.get_neighbourhood_array_3d(array_3d, (slc_in, row_in+radius, col_in), radius, {0}, pad), num_bins, (min_range, max_range)) #Include effect of new row
                            )

    elif neighbouring_dims == {0,2}:
        for (slc_out, slc_in) in enumerate(range(slc_slice.start, slc_slice.stop)):
            for (row_out, row_in) in enumerate(range(row_slice.start, row_slice.stop)):
                for (col_out, col_in) in enumerate(range(col_slice.start, col_slice.stop)):
                    if col_out == 0:
                        result[slc_out, row_out, col_out, :] = histogram(regions.get_neighbourhood_array_3d(array_3d, (slc_in, row_in, col_in), radius, {0,2}, pad), num_bins, (min_range, max_range)) #Get the only completely computed histograms (left column)
                    else:
                        result[slc_out, row_out, col_out, :] = (
                                result[slc_out, row_out, col_out-1, :]
                                -
                                histogram(regions.get_neighbourhood_array_3d(array_3d, (slc_in, row_in, col_in-1-radius), radius, {0}, pad), num_bins, (min_range, max_range)) #Undo effect of dropped column
                                +
                                histogram(regions.get_neighbourhood_array_3d(array_3d, (slc_in, row_in, col_in+radius), radius, {0}, pad), num_bins, (min_range, max_range)) #Include effect of new column
                            )

    elif neighbouring_dims == {1,2}:
        for (slc_out, slc_in) in enumerate(range(slc_slice.start, slc_slice.stop)):
            for (row_out, row_in) in enumerate(range(row_slice.start, row_slice.stop)):
                for (col_out, col_in) in enumerate(range(col_slice.start, col_slice.stop)):
                    if col_out == 0 and row_out == 0:
                        result[slc_out, row_out, col_out, :] = histogram(regions.get_neighbourhood_array_3d(array_3d, (slc_in, row_in, col_in), radius, {1,2}, pad), num_bins, (min_range, max_range)) #Get the only completely computed histogram (top left corner)
                    elif col_out == 0:
                        result[slc_out, row_out, col_out, :] = (
                                result[slc_out, row_out-1, col_out, :]
                                -
                                histogram(regions.get_neighbourhood_array_3d(array_3d, (slc_in, row_in-1-radius, col_in), radius, {2}, pad), num_bins, (min_range, max_range)) #Undo effect of dropped row
                                +
                                histogram(regions.get_neighbourhood_array_3d(array_3d, (slc_in, row_in+radius, col_in), radius, {2}, pad), num_bins, (min_range, max_range)) #Include effect of new row
                            )
                    else:
                        result[slc_out, row_out, col_out, :] = (
                                result[slc_out, row_out, col_out-1, :]
                                -
                                histogram(regions.get_neighbourhood_array_3d(array_3d, (slc_in, row_in, col_in-1-radius), radius, {1}, pad), num_bins, (min_range, max_range)) #Undo effect of dropped column
                                +
                                histogram(regions.get_neighbourhood_array_3d(array_3d, (slc_in, row_in, col_in+radius), radius, {1}, pad), num_bins, (min_range, max_range)) #Include effect of new column
                            )

    else:
        raise NotImplementedError('Only neighbouring dimensions of {0,1}, {0,2}, {1,2}, and {0,1,2} implemented.')

    return result

#########################################
def gpu_apply_histogram_to_all_neighbourhoods_in_slice_3d(array_3d, slice_range, radius, neighbouring_dims, min_range, max_range, num_bins, pad=0, row_slice=slice(None), col_slice=slice(None)):
    '''
    GPU implementation of apply_histogram_to_all_neighbourhoods_in_slice_3d.

    See apply_histogram_to_all_neighbourhoods_in_slice_3d for more information.

    :param numpy.ndarray array_3d: The volume from which to extract the histograms, expected to be of type uint16.
    :param slice slice_range: The range of slices to use within the volume.
    :param int radius: The radius of the neighbourhood around each voxel.
    :param set neighbouring_dims: The set of dimensions to apply the neighbourhoods on.
    :param int min_range: The minimum range of the values to consider.
    :param int max_range: The maximum range of the values to consider, not included.
    :param int num_bins: The number of bins in the histograms.
    :param int pad: The pad value for values outside the array.
    :param slice row_slice: The range of rows in the slice to consider.
    :param slice col_slice: The range of columns in the slice to consider.
    :return: A 3D array where the first two dimensions are equal to the dimensions of the slice
        and the last dimension is the number of bins.
    :rtype: numpy.ndarray
    '''

    if not cuda.gpu_available:
        raise ValueError('GPU is not available.')

    ## Example parameters
    # array_3d.shape = (23, 172, 202)
    # slice_range = 11
    # radius = 11
    # neighbouring_dims = {0, 1, 2}
    # min_range, max_range, num_bins = 0, 65536, 32
    # pad = 0
    # row_slice =  slice(11, 161, None)
    # col_slice = slice(11, 191, None)
    ## Output
    # result.shape = (150, 180, 32)

    ## Index ordering convention
    # outer to inner: slice, row, col = z, y, x

    # output histogram
    rows = row_slice.stop - row_slice.start
    cols = col_slice.stop - col_slice.start
    slices = slice_range.stop - slice_range.start
    result = np.zeros((slices, rows, cols, num_bins), dtype=featurisers.feature_dtype)

    # input volume
    assert array_3d.dtype == np.uint16
    NZ, NY, NX = array_3d.shape

    if neighbouring_dims == {0,1,2}:
        # GPU block size (working sizes)
        WS_X = 16
        WS_Y = 16
        # kernel parameters
        blocksize = (WS_X, WS_Y, 1)
        gridsize = ( int(math.ceil( float(cols) / float(WS_X) )),
                     int(math.ceil( float(rows) / float(WS_Y) )),
                     1 )
        sharedsize_tile = 1 * (2 * radius + WS_X) * (2 * radius + WS_Y) # index_t
        sharedsize_hist = 4 * WS_X * WS_Y * num_bins # result_t
        sharedsize = sharedsize_tile + sharedsize_hist
        # kernel call
        ch = cuda.histograms()
        ch.histogram_3d( cuda.drv.Out(result),
                    np.int32(min_range), np.int32(max_range), np.int32(num_bins),
                    cuda.drv.In(array_3d),
                    np.int32(NX), np.int32(NY), np.int32(NZ),
                    np.int32(col_slice.start), np.int32(col_slice.stop),
                    np.int32(row_slice.start), np.int32(row_slice.stop),
                    np.int32(slice_range.start), np.int32(slice_range.stop),
                    np.int32(radius),

                    block=blocksize,
                    grid=gridsize,
                    shared=sharedsize )
    elif neighbouring_dims == {0,1}: # y,z
        # GPU block size (working sizes)
        WS_Y = 16
        WS_Z = 16
        # kernel parameters
        blocksize = (1, WS_Y, WS_Z)
        gridsize = ( cols,
                     int(math.ceil( float(rows) / float(WS_Y) )),
                     int(math.ceil( float(slices) / float(WS_Z) ))
                     )
        sharedsize_tile = 1 * (2 * radius + WS_Y) * (2 * radius + WS_Z) # index_t
        sharedsize_hist = 4 * WS_Y * WS_Z * num_bins # result_t
        sharedsize = sharedsize_tile + sharedsize_hist
        # kernel call
        ch = cuda.histograms()
        ch.histogram_2d_yz( cuda.drv.Out(result),
                    np.int32(min_range), np.int32(max_range), np.int32(num_bins),
                    cuda.drv.In(array_3d),
                    np.int32(NX), np.int32(NY), np.int32(NZ),
                    np.int32(col_slice.start), np.int32(col_slice.stop),
                    np.int32(row_slice.start), np.int32(row_slice.stop),
                    np.int32(slice_range.start), np.int32(slice_range.stop),
                    np.int32(radius),

                    block=blocksize,
                    grid=gridsize,
                    shared=sharedsize )
    elif neighbouring_dims == {0,2}: # x,z
        # GPU block size (working sizes)
        WS_X = 16
        WS_Z = 16
        # kernel parameters
        blocksize = (WS_X, 1, WS_Z)
        gridsize = ( int(math.ceil( float(cols) / float(WS_X) )),
                     rows,
                     int(math.ceil( float(slices) / float(WS_Z) ))
                     )
        sharedsize_tile = 1 * (2 * radius + WS_X) * (2 * radius + WS_Z) # index_t
        sharedsize_hist = 4 * WS_X * WS_Z * num_bins # result_t
        sharedsize = sharedsize_tile + sharedsize_hist
        # kernel call
        ch = cuda.histograms()
        ch.histogram_2d_xz( cuda.drv.Out(result),
                    np.int32(min_range), np.int32(max_range), np.int32(num_bins),
                    cuda.drv.In(array_3d),
                    np.int32(NX), np.int32(NY), np.int32(NZ),
                    np.int32(col_slice.start), np.int32(col_slice.stop),
                    np.int32(row_slice.start), np.int32(row_slice.stop),
                    np.int32(slice_range.start), np.int32(slice_range.stop),
                    np.int32(radius),

                    block=blocksize,
                    grid=gridsize,
                    shared=sharedsize )
    elif neighbouring_dims == {1,2}: # x,y
        # GPU block size (working sizes)
        WS_X = 16
        WS_Y = 16
        # kernel parameters
        blocksize = (WS_X, WS_Y, 1)
        gridsize = ( int(math.ceil( float(cols) / float(WS_X) )),
                     int(math.ceil( float(rows) / float(WS_Y) )),
                     slices )
        sharedsize_tile = 1 * (2 * radius + WS_X) * (2 * radius + WS_Y) # index_t
        sharedsize_hist = 4 * WS_X * WS_Y * num_bins # result_t
        sharedsize = sharedsize_tile + sharedsize_hist
        # kernel call
        ch = cuda.histograms()
        ch.histogram_2d_xy( cuda.drv.Out(result),
                    np.int32(min_range), np.int32(max_range), np.int32(num_bins),
                    cuda.drv.In(array_3d),
                    np.int32(NX), np.int32(NY), np.int32(NZ),
                    np.int32(col_slice.start), np.int32(col_slice.stop),
                    np.int32(row_slice.start), np.int32(row_slice.stop),
                    np.int32(slice_range.start), np.int32(slice_range.stop),
                    np.int32(radius),

                    block=blocksize,
                    grid=gridsize,
                    shared=sharedsize )
    else:
        raise NotImplementedError('Only neighbouring dimensions of {0,1}, {0,2}, {1,2}, and {0,1,2} implemented.')

    return result
