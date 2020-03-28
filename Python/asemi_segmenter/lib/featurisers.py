'''Module containing different methods to turn voxels in a volume into feature vectors.'''

import numpy as np
import random
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
    def get_config(self):
        '''
        Get the dictionary configuration of the featuriser's parameters.
        
        :return: The dictionary configuration.
        :rtype: dict
        '''
        raise NotImplementedError()
    
    #########################################
    def _fix_ranges(self, data_scales, row_range, col_range):
        '''
        (Protected method) Replace None in the row and column ranges with actual numbers.
        
        :param dict data_scales: A dictionary of different scales of the volume.
        :param slice row_range: The range of rows to featurise (if only a part of the slice is
            needed).
        :param slice col_range: The range of columns to featurise (if only a part of the slice is
            needed).
        :return: Tuple with ranges fixed (row_range, col_range)
        :rtype: tuple
        '''
        row_range = slice(row_range.start if row_range.start is not None else 0, row_range.stop if row_range.stop is not None else data_scales[0].shape[1])
        col_range = slice(col_range.start if col_range.start is not None else 0, col_range.stop if col_range.stop is not None else data_scales[0].shape[2])
        return (row_range, col_range)
        
    #########################################
    def _prepare_featurise(self, data_scales, row_range, col_range, output, output_start_row_index, output_start_col_index):
        '''
        (Protected method) Fix any missing parameters sent to featurise method.
        
        :param dict data_scales: A dictionary of different scales of the volume.
        :param slice row_range: The range of rows to featurise (if only a part of the slice is
            needed).
        :param slice col_range: The range of columns to featurise (if only a part of the slice is
            needed).
        :param numpy.ndarray output: The output dataset to contain the features (will be created
            if None). Must be 2D.
        :param int output_start_row_index: The row index in the output dataset to start putting
            feature vectors in.
        :param int output_start_col_index: The column index in the output dataset to start putting
            feature vectors from.
        :return: Tuple with fixed parameters (rows_needed, cols_needed, row_range, col_range,
            output).
        :rtype: tuple
        '''
        feature_size = self.get_feature_size()
        
        (row_range, col_range) = self._fix_ranges(data_scales, row_range, col_range)
        
        if output is None:
            output = np.empty(((row_range.stop-row_range.start)*(col_range.stop-col_range.start), feature_size), np.float32)
        if len(output.shape) != 2 or output.dtype != np.float32:
            raise ValueError('Output array must be a float32 matrix.')
        
        rows_needed = (row_range.stop - row_range.start)*(col_range.stop - col_range.start)
        last_output_row_index = output_start_row_index + rows_needed - 1
        if last_output_row_index >= output.shape[0]:
            raise ValueError('Provided output array does not have enough rows to hold result in expected range (array rows = {}, rows needed = {}, last output row index = {}).'.format(output.shape[0], rows_needed, last_output_row_index))
        
        cols_needed = feature_size
        last_output_col_index = output_start_col_index + cols_needed - 1
        if last_output_col_index >= output.shape[1]:
            raise ValueError('Provided output array does not have enough columns to hold result in expected range (array columns = {}, columns needed = {}, last output column index = {}).'.format(output.shape[1], cols_needed, last_output_col_index))
        
        return (rows_needed, cols_needed, row_range, col_range, output)
    
    #########################################
    def featurise(self, data_scales, slice_index, block_rows, block_cols, row_range=slice(None), col_range=slice(None), output=None, output_start_row_index=0, output_start_col_index=0, n_jobs=1):
        '''
        Turn a slice from a volume into a matrix of feature vectors.

        The matrix will have a row for every voxel in the slice and the number of columns is the
        feature size. This function is made with creating a dataset in mind, that is, where the
        feature vectors are to be presented as a list rather than in the same shape of the slice.
        In order to be more memory efficient, the result can be placed in a reserved dataset
        matrix that should contain the features of a number of slices. The position of the current
        slice's features within this larger dataset can controlled using the output_start_row_index
        and output_start_col_index parameters.
        
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
        :param int output_start_row_index: The row index in the output dataset to start putting
            feature vectors in.
        :param int output_start_col_index: The column index in the output dataset to start putting
            feature vectors from.
        :param int n_jobs: The number of concurrent processes to use.
        :return: A reference to output.
        :rtype: numpy.ndarray
        '''
        raise NotImplementedError()


#########################################
class VoxelFeaturiser(Featuriser):
    '''Just treat the voxel values as features.'''
    
    #########################################
    def __init__(self):
        '''
        Constructor.
        '''
        pass
    
    #########################################
    def get_feature_size(self):
        '''
        Get the number of elements in the feature vector.
        
        :return: The feature vector size.
        :rtype: int
        '''
        return 1
    
    #########################################
    def get_context_needed(self):
        '''
        Get the maximum amount of context needed around a voxel to generate a feature vector.
        
        :return: The context size.
        :rtype: int
        '''
        return 0
    
    #########################################
    def get_scales_needed(self):
        '''
        Get the different volume scales needed to generate a feature vector.
        
        :return: The scales needed.
        :rtype: set
        '''
        return { 0 }
    
    #########################################
    def get_config(self):
        '''
        Get the dictionary configuration of the featuriser's parameters.
        
        :return: The dictionary configuration.
        :rtype: dict
        '''
        return {'type': 'voxel', 'params': dict()}
    
    #########################################
    def featurise(self, data_scales, slice_index, block_rows, block_cols, row_range=slice(None), col_range=slice(None), output=None, output_start_row_index=0, output_start_col_index=0, n_jobs=1):
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
        :param int output_start_row_index: As described in the super class.
        :param int output_start_col_index: As described in the super class.
        :param int n_jobs: As described in the super class.
        :param callable progress_listener: As described in the super class.
        :return: As described in the super class.
        :rtype: numpy.ndarray
        '''
        (output_rows_needed, output_cols_needed, row_range, col_range, output) = self._prepare_featurise(data_scales, row_range, col_range, output, output_start_row_index, output_start_col_index)
        
        output[
            output_start_row_index:output_start_row_index+output_rows_needed,
            output_start_col_index:output_start_col_index+output_cols_needed
            ] = np.reshape(data_scales[0][slice_index, row_range, col_range], (-1, 1))
        return output


#########################################
class HistogramFeaturiser(Featuriser):
    '''
    A featuriser that collects a histogram of voxel values around each voxel.
    
    The feature vector of a voxel consists of histograms of cubes of a given sizes and at a given
    scale centered on the voxel. The number of bins in each histogram is controllable but the range
    of the bins is always uniformly divided between 0 and 2^16-1.
    '''
    
    #########################################
    @classmethod
    def create_random(cls, num_scales_available, rand=random.Random()):
        '''
        Create a random histogram featuriser object.
        
        :return: The featuriser object.
        :rtype: HistogramFeaturiser
        '''
        return HistogramFeaturiser(
            rand.randrange(1, 128+1),
            rand.randrange(0, num_scales_available+1),
            rand.randrange(2, 64+1),
            )
    
    #########################################
    def __init__(self, radius, scale, num_bins):
        '''
        Constructor.
        
        :param int radius: The neighbourhood radius.
        :param int scale: The scale of the volume from which to extract this neighbourhood.
        :param int num_bins: num_bins is the number of bins in the histogram.
        '''
        self.radius = radius
        self.scale = scale
        self.num_bins = num_bins
        
    #########################################
    def get_feature_size(self):
        '''
        Get the number of elements in the feature vector.
        
        :return: The feature vector size.
        :rtype: int
        '''
        return self.num_bins
    
    #########################################
    def get_context_needed(self):
        '''
        Get the maximum amount of context needed around a voxel to generate a feature vector.
        
        :return: The context size.
        :rtype: int
        '''
        return self.radius
    
    #########################################
    def get_scales_needed(self):
        '''
        Get the different volume scales needed to generate a feature vector.
        
        :return: The scales needed.
        :rtype: set
        '''
        return { 0, self.scale }
    
    #########################################
    def get_config(self):
        '''
        Get the dictionary configuration of the featuriser's parameters.
        
        :return: The dictionary configuration.
        :rtype: dict
        '''
        return {'type': 'histogram', 'params': {'radius': self.radius, 'scale': self.scale, 'num_bins': self.num_bins}}
    
    #########################################
    def featurise(self, data_scales, slice_index, block_rows, block_cols, row_range=slice(None), col_range=slice(None), output=None, output_start_row_index=0, output_start_col_index=0, n_jobs=1):
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
        :param int output_start_row_index: As described in the super class.
        :param int output_start_col_index: As described in the super class.
        :param int n_jobs: As described in the super class.
        :param callable progress_listener: As described in the super class.
        :return: As described in the super class.
        :rtype: numpy.ndarray
        '''
        
        def processor(params, radius, scale, num_bins, full_input_ranges, output_start_row_index, output_start_col_index):
            '''Processor for process_array_in_blocks_single_slice.'''
            [ num_rows_out, num_cols_out ] = params[0]['contextless_shape']
            
            hists = histograms.apply_histogram_to_all_neighbourhoods_in_slice_3d(
                params[scale]['block'],
                params[scale]['contextless_slices_wrt_block'][0],
                radius,
                {0,1,2},
                0, 2**16,
                num_bins,
                row_slice=params[scale]['contextless_slices_wrt_block'][1], col_slice=params[scale]['contextless_slices_wrt_block'][2]
                )
            features = np.reshape(downscales.grow_array(hists, scale, [0, 1], params[0]['contextless_shape']), (-1, num_bins)).astype(np.float32)
            
            out_indexes = (
                    [
                        output_start_row_index + row*(full_input_ranges[1].stop - full_input_ranges[1].start) + col
                        for row in range(params[0]['contextless_slices_wrt_range'][1].start, params[0]['contextless_slices_wrt_range'][1].stop)
                        for col in range(params[0]['contextless_slices_wrt_range'][2].start, params[0]['contextless_slices_wrt_range'][2].stop)
                    ],
                    slice(output_start_col_index, output_start_col_index+num_bins)
                )
            return (features, out_indexes)
        
        (output_rows_needed, output_cols_needed, row_range, col_range, output) = self._prepare_featurise(data_scales, row_range, col_range, output, output_start_row_index, output_start_col_index)
        
        return arrayprocs.process_array_in_blocks_single_slice(
            data_scales,
            output,
            processor,
            block_shape=(block_rows, block_cols),
            slice_index=slice_index,
            scales=sorted({0, self.scale}),
            in_ranges=[row_range, col_range],
            context_size=self.radius,
            n_jobs=n_jobs,
            extra_params=(self.radius, self.scale, self.num_bins, (row_range, col_range), output_start_row_index, output_start_col_index),
            )


#########################################
class CompositeFeaturiser(Featuriser):
    '''Combine several featurisers into one with the feature vectors being concatenated.'''
    
    #########################################
    @classmethod
    def create_random(cls, max_num_histograms, num_scales_available, rand=random.Random()):
        '''
        Create a random composite featuriser object.
        
        :return: The featuriser object.
        :rtype: CompositeFeaturiser
        '''
        featuriser_list = []
        if rand.choice([True, False]):
            featuriser_list.append(VoxelFeaturiser())
        for _ in range(rand.randrange(1, max_num_histograms+1)):
            featuriser_list.append(HistogramFeaturiser.create_random(num_scales_available, rand))
        return CompositeFeaturiser(featuriser_list)
    
    #########################################
    def __init__(self, featuriser_list):
        '''
        Constructor.
        
        :param list featuriser_list: List of featuriser objects to combine.
        '''
        self.featuriser_list = featuriser_list
    
    #########################################
    def get_feature_size(self):
        '''
        Get the number of elements in the feature vector.
        
        :return: The feature vector size.
        :rtype: int
        '''
        return sum(featuriser.get_feature_size() for featuriser in self.featuriser_list)
    
    #########################################
    def get_context_needed(self):
        '''
        Get the maximum amount of context needed around a voxel to generate a feature vector.
        
        :return: The context size.
        :rtype: int
        '''
        return max(featuriser.get_context_needed() for featuriser in self.featuriser_list)
    
    #########################################
    def get_scales_needed(self):
        '''
        Get the different volume scales needed to generate a feature vector.
        
        :return: The scales needed.
        :rtype: set
        '''
        return set.union(*(featuriser.get_scales_needed() for featuriser in self.featuriser_list))
    
    #########################################
    def get_config(self):
        '''
        Get the dictionary configuration of the featuriser's parameters.
        
        :return: The dictionary configuration.
        :rtype: dict
        '''
        return {'type': 'composite', 'params': {'featuriser_list': [sub_featuriser.get_config() for sub_featuriser in self.featuriser_list]}}
    
    #########################################
    def featurise(self, data_scales, slice_index, block_rows, block_cols, row_range=slice(None), col_range=slice(None), output=None, output_start_row_index=0, output_start_col_index=0, n_jobs=1):
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
        :param int output_start_row_index: As described in the super class.
        :param int output_start_col_index: As described in the super class.
        :param int n_jobs: As described in the super class.
        :return: As described in the super class.
        :rtype: numpy.ndarray
        '''
        (output_rows_needed, output_cols_needed, row_range, col_range, output) = self._prepare_featurise(data_scales, row_range, col_range, output, output_start_row_index, output_start_col_index)
        
        for featuriser in self.featuriser_list:
            featuriser.featurise(data_scales, slice_index, block_rows, block_cols, row_range, col_range, output, output_start_row_index, output_start_col_index, n_jobs)
            output_start_col_index += featuriser.get_feature_size()
        
        return output