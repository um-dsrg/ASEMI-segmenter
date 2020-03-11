import os
import numpy as np
import fast_histogram
import sys
from asemi_segmenter.lib import regions

#########################################
def histogram(array, num_bins, value_range):
    '''value_range = (min, max_not_inclusive)'''
    return fast_histogram.histogram1d(array, num_bins, value_range)
    #return np.histogram(array, num_bins, value_range)[0] #Some times fast_histogram fails test cases because of a minor bug (https://github.com/astrofrog/fast-histogram/issues/45) so this may be useful during testing.

#########################################
def apply_histogram_to_all_neighbourhoods_in_slice_3d(array_3d, slice_index, radius, neighbouring_dims, min_range, max_range, num_bins, pad=0, row_slice=slice(None), col_slice=slice(None)):
    '''
    max_range not included.
    
    Histograms of neighbourhoods around every voxel in a slice can be computed efficiently using a rolling algorithm that reuses histograms in other neighbouring voxels.
    Consider the following slice:
        [a,b,c]
        [d,e,f]
        [g,h,i]
    Then the neighbourhood around (0,0) with radius 1 has the following frequencies:
        PAD=5, a=1, b=1, c=0, d=1, e=1, f=0, g=0, h=0, i=0 => [5,1,1,0,1,1,0,0,0,0]
    The neighbourhood around (0,1) has values in common with the neighbourhood around (0,0) so we can avoid counting all the elements in this neighbourhood by instead counting only what has changed from the previous neighbourhood and update the frequencies with new information in the dropped and new columns:
        histogram_01 = histogram_00 - histogram([PAD,a,d]) + histogram([PAD,c,f])
    Likewise, the neighbourhood around (1,0) has values in common with (0,0) as well and can be calculated by subtracting the dropped row and adding the new column:
        histogram_10 = histogram_00 - histogram([PAD,PAD,PAD]) + histogram([g,h,i])
    This means that we only need to perform a full histogram count for the top left corner as everything else can be computed by rolling the values.
    
    For extracting histograms in 3D space, we will still be only processing a single slice but with adjacent slices for context. The neighbouring_dims orientation of the neighbourhood will change what can be reused however.
    Given a 3D array A, the neighbourhood around A[r,c] (in the slice of interest) with radius R and neighbouring_dims d is hist(A[r,c], R, d). For each d, the following optimisations can be performed:
        hist(A[r,c], R, [1,2,3]) = hist(A[r-1,c], R, [1,2,3]) - hist(A[r-1-R,c], R, [1,3]) + hist(A[r+R,c], R, [1,3])
                          (also) = hist(A[r,c-1], R, [1,2,3]) - hist(A[r,c-1-R], R, [1,2]) + hist(A[r,c+R], R, [1,2])
    '''
    [ _, num_rows_in, num_cols_in ] = array_3d.shape
    row_slice = slice(
            row_slice.start if row_slice.start is not None else 0,
            row_slice.stop  if row_slice.stop is not None else num_rows_in
        )
    col_slice = slice(
            col_slice.start if col_slice.start is not None else 0,
            col_slice.stop  if col_slice.stop is not None else num_cols_in
        )
    num_rows_out = row_slice.stop - row_slice.start
    num_cols_out = col_slice.stop - col_slice.start
    
    result = np.empty([ num_rows_out, num_cols_out, num_bins ], np.float32)
    
    if neighbouring_dims == {1,2,3}:
        for (row_out, row_in) in enumerate(range(row_slice.start, row_slice.stop)):
            for (col_out, col_in) in enumerate(range(col_slice.start, col_slice.stop)):
                if col_out == 0 and row_out == 0:
                    result[row_out, col_out, :] = histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in, col_in), radius, {1,2,3}, pad), num_bins, (min_range, max_range)) #Get the only completely computed histogram (top left corner).
                elif col_out == 0:
                    result[row_out, col_out, :] = (
                            result[row_out-1, col_out, :]
                            -
                            histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in-1-radius, col_in), radius, {1,3}, pad), num_bins, (min_range, max_range)) #Undo effect of dropped row face.
                            +
                            histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in+radius, col_in), radius, {1,3}, pad), num_bins, (min_range, max_range)) #Include effect of new row face.
                        )
                else:
                    result[row_out, col_out, :] = (
                            result[row_out, col_out-1, :]
                            -
                            histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in, col_in-1-radius), radius, {1,2}, pad), num_bins, (min_range, max_range)) #Undo effect of dropped column face.
                            +
                            histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in, col_in+radius), radius, {1,2}, pad), num_bins, (min_range, max_range)) #Include effect of new column face.
                        )
    else:
        raise NotImplementedError('Only neighbouring dimensions of {1,2,3} implemented.')
        
    return result
