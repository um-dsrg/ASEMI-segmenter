'''Module containing different methods to turn voxels in a volume into feature vectors.'''

import numpy as np
import os
import sys
from asemi_segmenter.lib import regions
from asemi_segmenter.lib import histograms
from asemi_segmenter.lib import downscales
from asemi_segmenter.lib import arrayprocs

#########################################
class Featuriser(object):
    '''Super class for featurisers.'''
    
    #########################################
    def __init__(self):
        '''Empty constructor.'''
        pass
    
    #########################################
    def get_feature_size(self):
        '''
        Get the number of elements in the feature vector.
        
        :return: The feature vector size.
        :rtype: int
        '''
        raise NotImplementedError()
    
    #########################################
    def get_context_needed(self):
        '''
        Get the maximum amount of context needed around a voxel to generate a feature vector.
        
        :return: The context size.
        :rtype: int
        '''
        raise NotImplementedError()
    
    #########################################
    def get_scales_needed(self):
        '''
        Get the different volume scales needed to generate a feature vector.
        
        :return: The scales needed.
        :rtype: set
        '''
        raise NotImplementedError()
    
    #########################################
    def featurise(self, data_scales, slice_index, block_rows, block_cols, row_range=slice(None), col_range=slice(None), output=None, output_start_index=0, n_jobs=1, progress_listener=lambda num_ready, num_new:None):
        '''
        Turn a slice from a volume into a matrix of feature vectors.

        The matrix will have a row for every voxel in the slice and the number of columns is the
        feature size. This function is made with creating a dataset in mind, that is, where the
        feature vectors are to be presented as a list rather than in the same shape of the slice.
        In order to be more memory efficient, the result can be placed in a reserved dataset
        matrix that should contain the features of a number of slices. The position of the current
        slice's features within this larger dataset can controlled using the output_start_index
        parameter.
        
        :param dict data_scales: A dictionary of different scales of the volume.
        :param int slice_index: The index of the slice in the volume to featurise.
        :param int block_rows: The first dimension of the block shape (2D).
        :param int block_cols: The second dimension of the block shape (2D).
        :param slice row_range: The range of rows to featurise (if only a part of the slice is
            needed).
        :param slice col_range: The range of columns to featurise (if only a part of the slice is
            needed).
        :param numpy.ndarray output: The output dataset to contain the features (will be created
            if None). Must be 2D.
        :param int output_start_index: The row index in the output dataset to start putting
            feature vectors in.
        :param int n_jobs: The number of concurrent processes to use.
        :param callable progress_listener: A function that receives information about the progress
            of the downscale.
        '''
        raise NotImplementedError()

#########################################
class HistogramFeaturiser(Featuriser):
    '''
    A featuriser that collects histograms of voxel values around each voxel.
    
    The feature vector of a voxel consists of the voxel value itself, followed by a number of
    histograms of squares of different sizes and at different scales centered on the voxel. The
    number of bins in each histogram is controllable but the range of the bins is always uniformly
    divided between 0 and 2^16-1.
    
    Feature vector plan:
        For each voxel in slice [
            if use_voxel_value:
                voxel value (1 element)
            for (radius, scale, num_bins) in histogram_params:
                histogram of values in cube neighbourhood around voxel (num_bins elements)
        ]
    '''
    
    #########################################
    def __init__(self, use_voxel_value, histogram_params):
        '''
        Constructor.
        
        :param bool use_voxel_value: Whether to also include 
        :param list histogram_params: A list of parameters for each histogram consisting of the
            triple (radius, scale, num_bins) where radius is the radius of the square
            neighbourhood around the voxel (such that the side of the square is radius+1+radius
            long), scale is the scale of the volume from which to extract this neighbourhood, and
            num_bins is the number of bins in the histogram.
        '''
        self.use_voxel_value = use_voxel_value
        self.histogram_params = histogram_params
        self.feature_size = 1*self.use_voxel_value + sum(num_bins for (radius, scale, num_bins) in self.histogram_params)
        self.context_needed = max(radius for (radius, scale, num_bins) in self.histogram_params)
        
    #########################################
    def get_feature_size(self):
        '''
        Get the number of elements in the feature vector.
        
        :return: The feature vector size.
        :rtype: int
        '''
        return self.feature_size
    
    #########################################
    def get_context_needed(self):
        '''
        Get the maximum amount of context needed around a voxel to generate a feature vector.
        
        :return: The context size.
        :rtype: int
        '''
        return self.context_needed
    
    #########################################
    def get_scales_needed(self):
        '''
        Get the different volume scales needed to generate a feature vector.
        
        :return: The scales needed.
        :rtype: set
        '''
        return { 0 } | { scale for (radius, scale, num_bins) in self.histogram_params }
    
    #########################################
    def featurise(self, data_scales, slice_index, block_rows, block_cols, row_range=slice(None), col_range=slice(None), output=None, output_start_index=0, n_jobs=1, progress_listener=lambda num_ready, num_new:None):
        '''
        Turn a slice from a volume into a matrix of feature vectors.

        See super class for more information.
        
        :param dict data_scales: As described in the super class.
        :param int slice_index: As described in the super class.
        :param int block_rows: As described in the super class.
        :param int block_cols: As described in the super class.
        :param slice row_range: As described in the super class.
        :param slice col_range: As described in the super class.
        :param numpy.ndarray output: As described in the super class.
        :param int output_start_index: As described in the super class.
        :param int n_jobs: As described in the super class.
        :param callable progress_listener: As described in the super class.
        '''
        
        def processor(params, use_voxel_value, histogram_params, feature_size, full_input_ranges, output_start_index):
            '''Processor for process_array_in_blocks_single_slice.'''
            [ num_rows_out, num_cols_out ] = params[0]['contextless_shape']
            
            features = np.empty([ num_rows_out, num_cols_out, feature_size ], np.float32)
            feature_pos = 0
            
            if use_voxel_value:
                #center voxel value
                features[:, :, feature_pos] = params[0]['block'][params[0]['contextless_slices_wrt_block']]
                feature_pos += 1
            
            for (radius, scale, num_bins) in histogram_params:
                hists = histograms.apply_histogram_to_all_neighbourhoods_in_slice_3d(
                        params[scale]['block'],
                        params[scale]['contextless_slices_wrt_block'][0],
                        radius,
                        {1,2,3},
                        0, 2**16,
                        num_bins,
                        row_slice=params[scale]['contextless_slices_wrt_block'][1], col_slice=params[scale]['contextless_slices_wrt_block'][2]
                    )
                hists = downscales.grow_array(hists, scale, [0, 1], params[0]['contextless_shape'])
                
                #cube value histogram
                features[:, :, feature_pos:feature_pos+num_bins] = hists
                
                feature_pos += num_bins
            
            out_indexes = (
                    [
                        output_start_index + row*(full_input_ranges[1].stop - full_input_ranges[1].start) + col
                        for row in range(params[0]['contextless_slices_wrt_range'][1].start, params[0]['contextless_slices_wrt_range'][1].stop)
                        for col in range(params[0]['contextless_slices_wrt_range'][2].start, params[0]['contextless_slices_wrt_range'][2].stop)
                    ],
                    slice(None)
                )
            return (features.reshape([-1, feature_size]), out_indexes)
        
        row_range = slice(row_range.start if row_range.start is not None else 0, row_range.stop if row_range.stop is not None else data_scales[0].shape[1])
        col_range = slice(col_range.start if col_range.start is not None else 0, col_range.stop if col_range.stop is not None else data_scales[0].shape[2])
        
        if output is None:
            output = np.empty(((row_range.stop-row_range.start)*(col_range.stop-col_range.start), self.feature_size), np.float32)
        if len(output.shape) != 2 or output.dtype != np.float32:
            raise ValueError('Output array must be a float32 matrix.')
        if output.shape[1] != self.feature_size:
            raise ValueError('Output array is not wide enough to hold feature vectors (expected={}, provided={}).'.format(self.feature_size, output.shape[1]))
        
        final_output_size = (row_range.stop - row_range.start)*(col_range.stop - col_range.start)
        post_output_index = output_start_index + final_output_size
        last_output_index = post_output_index - 1
        if last_output_index >= output.shape[0]:
            raise ValueError('Provided output array is not big enough to hold result in expected range (array size = {}, expected output size = {}, last output index = {}).'.format(output.shape[0], final_output_size, last_output_index))
        
        return (
                arrayprocs.process_array_in_blocks_single_slice(
                    data_scales,
                    output,
                    processor,
                    block_shape=[block_rows, block_cols],
                    slice_index=slice_index,
                    scales=sorted({ 0 } | { scale for (radius, scale, num_bins) in self.histogram_params }),
                    in_ranges=[ row_range, col_range ],
                    context_size=self.context_needed,
                    n_jobs=n_jobs,
                    extra_params=(self.use_voxel_value, self.histogram_params, self.feature_size, (row_range, col_range), output_start_index),
                    progress_listener=progress_listener,
                ),
                post_output_index
            )