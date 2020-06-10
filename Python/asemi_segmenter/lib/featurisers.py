'''Module containing different methods to turn voxels in a volume into feature vectors.'''

import json
import numpy as np
import skimage.feature
import h5py
from asemi_segmenter.lib import histograms
from asemi_segmenter.lib import downscales
from asemi_segmenter.lib import arrayprocs
from asemi_segmenter.lib import regions
from asemi_segmenter.lib import validations
from asemi_segmenter.lib import samplers


#########################################
def load_featuriser_from_config(config, sampler_factory=None, use_gpu=False):
    '''
    Load a featuriser from a configuration dictionary.

    :param dict config: Configuration of the featuriser.
    :param samplers.SamplerFactory sampler_factory: The factory to use to create samplers
        for the featuriser parameters. If None then only constant parameters can be used.
    :param bool use_gpu: Whether to use the GPU for computing features.
    :return: A featuriser object.
    :rtype: Featuriser
    '''
    validations.validate_json_with_schema_file(config, 'featuriser.json')

    def orientation_to_neighbouringdims(orientation):
        '''Convert a config orientation string into a neighbouring dims set.'''
        return {
            'yz': {0, 1},
            'right': {0, 1},
            'sagittal': {0, 1},

            'xz': {0, 2},
            'flat': {0, 2},
            'transverse': {0, 2},

            'xy': {1, 2},
            'front': {1, 2},
            'coronal': {1, 2},
            }[orientation]

    def recursive(config):
        '''Recursively read a nested configuration to produce a featuriser object.'''
        if config['type'] == 'voxel':
            return VoxelFeaturiser()

        elif config['type'] == 'histogram':
            radius = None
            scale = None
            num_bins = None

            if sampler_factory is not None:
                if isinstance(config['params']['radius'], dict):
                    radius = sampler_factory.create_integer_sampler(
                        config['params']['radius']['min'],
                        config['params']['radius']['max'],
                        config['params']['radius']['distribution']
                        )
                elif isinstance(config['params']['radius'], str):
                    radius = sampler_factory.get_named_sampler(
                        config['params']['radius'],
                        'integer'
                        )
                else:
                    radius = sampler_factory.create_constant_sampler(
                        config['params']['radius']
                        )
            else:
                if isinstance(config['params']['radius'], dict):
                    raise ValueError('radius must be a constant not a range.')
                radius = config['params']['radius']

            if sampler_factory is not None:
                if isinstance(config['params']['scale'], dict):
                    scale = sampler_factory.create_integer_sampler(
                        config['params']['scale']['min'],
                        config['params']['scale']['max'],
                        config['params']['scale']['distribution']
                        )
                elif isinstance(config['params']['scale'], str):
                    scale = sampler_factory.get_named_sampler(
                        config['params']['scale'],
                        'integer'
                        )
                else:
                    scale = sampler_factory.create_constant_sampler(
                        config['params']['scale']
                        )
            else:
                if isinstance(config['params']['scale'], dict):
                    raise ValueError('scale must be a constant not a range.')
                scale = config['params']['scale']

            if sampler_factory is not None:
                if isinstance(config['params']['num_bins'], dict):
                    num_bins = sampler_factory.create_integer_sampler(
                        config['params']['num_bins']['min'],
                        config['params']['num_bins']['max'],
                        config['params']['num_bins']['distribution']
                        )
                elif isinstance(config['params']['num_bins'], str):
                    num_bins = sampler_factory.get_named_sampler(
                        config['params']['num_bins'],
                        'integer'
                        )
                else:
                    num_bins = sampler_factory.create_constant_sampler(
                        config['params']['num_bins']
                        )
            else:
                if isinstance(config['params']['num_bins'], dict):
                    raise ValueError('num_bins must be a constant not a range.')
                num_bins = config['params']['num_bins']

            return HistogramFeaturiser(radius, scale, num_bins, use_gpu)

        elif config['type'] == 'lbp':
            neighbouring_dims = orientation_to_neighbouringdims(config['params']['orientation'])
            radius = None
            scale = None

            if sampler_factory is not None:
                if isinstance(config['params']['radius'], dict):
                    radius = sampler_factory.create_integer_sampler(
                        config['params']['radius']['min'],
                        config['params']['radius']['max'],
                        config['params']['radius']['distribution']
                        )
                elif isinstance(config['params']['radius'], str):
                    radius = sampler_factory.get_named_sampler(
                        config['params']['radius'],
                        'integer'
                        )
                else:
                    radius = sampler_factory.create_constant_sampler(
                        config['params']['radius']
                        )
            else:
                if isinstance(config['params']['radius'], dict):
                    raise ValueError('radius must be a constant not a range.')
                radius = config['params']['radius']

            if sampler_factory is not None:
                if isinstance(config['params']['scale'], dict):
                    scale = sampler_factory.create_integer_sampler(
                        config['params']['scale']['min'],
                        config['params']['scale']['max'],
                        config['params']['scale']['distribution']
                        )
                elif isinstance(config['params']['scale'], str):
                    scale = sampler_factory.get_named_sampler(
                        config['params']['scale'],
                        'integer'
                        )
                else:
                    scale = sampler_factory.create_constant_sampler(
                        config['params']['scale']
                        )
            else:
                if isinstance(config['params']['scale'], dict):
                    raise ValueError('scale must be a constant not a range.')
                scale = config['params']['scale']

            return LocalBinaryPatternFeaturiser(neighbouring_dims, radius, scale)

        elif config['type'] == 'composite':
            return CompositeFeaturiser(
                [recursive(sub_config) for sub_config in config['params']['featuriser_list']]
                )

        else:
            raise NotImplementedError(
                'Featuriser {} is not implemented.'.format(config['type'])
                )

    return recursive(config)


#########################################
class FeaturesTable(object):
    '''Lookup table to speed up featurisation.'''

    #########################################
    def __init__(self, data_fullfname=None):
        '''
        Constructor.

        :param str data_fullfname: Full file name (with path) to table HDF file.
            If None then lookup table will be kept in memory.
        '''
        self.data_fullfname = data_fullfname
        self.data = None

    #########################################
    def create(self):
        '''Create a new table.'''
        if self.data_fullfname is not None:
            with h5py.File(self.data_fullfname, 'w') as data_f:
                pass
        else:
            self.data = dict()

    #########################################
    def load(self):
        '''Load the HDF file.'''
        if self.data_fullfname is not None:
            self.data = h5py.File(self.data_fullfname, 'r+')

    #########################################
    def add_features(self, featuriser_id, featuriser_params, dataset_name, features):
        '''
        Add new features to HDF file.

        :param str featuriser_id: Unique name of the featuriser class.
        :param tuple featuriser_params: The tuple from the featuriser's get_params.
        :param str dataset_name: Unique name of the set of voxels that were featurised.
        :param numpy.ndarray features: The feature set to store (2D, float32).
        '''
        if len(features.shape) != 2:
            raise ValueError('Features must be 2 dimensional.')
        if features.dtype != np.float32:
            raise ValueError('Features must be type float32.')

        params = json.dumps(featuriser_params)
        if self.data_fullfname is not None:
            try:
                g = self.data[featuriser_id]
            except KeyError as ex:
                g = self.data.create_group(featuriser_id)

            try:
                g = g[params]
            except KeyError as ex:
                g = g.create_group(params)

            g.create_dataset(dataset_name, data=features, chunks=None)
        else:
            if featuriser_id not in self.data:
                self.data[featuriser_id] = dict()
            g = self.data[featuriser_id]

            if params not in g:
                g[params] = dict()
            g = g[params]
            g[dataset_name] = features

    #########################################
    def get_features(self, featuriser_id, featuriser_params, dataset_name):
        '''
        Get features from HDF file.

        :param str featuriser_id: Unique name of the featuriser class.
        :param tuple featuriser_params: The tuple from the featuriser's get_params.
        :param str dataset_name: Unique name of the set of voxels that were featurised.
        '''
        try:
            return self.data[featuriser_id][json.dumps(featuriser_params)][dataset_name]
        except KeyError as ex:
            return None

    #########################################
    def close(self):
        '''Close the HDF file.'''
        if self.data_fullfname is not None:
            self.data.close()
        self.data = None


#########################################
class Featuriser(object):
    '''Super class for featurisers.'''

    #########################################
    def __init__(self):
        '''Empty constructor.'''
        pass

    #########################################
    def refresh_parameters(self):
        '''
        Refresh parameter values from the samplers provided.
        '''
        raise NotImplementedError()

    #########################################
    def set_sampler_values(self, config):
        '''
        Set the values of the samplers provided according to a config.

        :param dict config: The configuration dictionary for the feature parameters.
        '''
        raise NotImplementedError()

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
    def get_params(self):
        '''
        Get the featuriser's parameters as nested tuples.

        :return: The parameters.
        :rtype: tuple
        '''
        raise NotImplementedError()

    #########################################
    def _fix_range(self, data_scales, dim, rng):
        '''
        (Protected method) Replace None in a range with actual numbers.

        :param dict data_scales: A dictionary of different scales of the volume.
        :param int dim: The dimension (in the data) of the range being fixed.
        :param slice rng: The range of data.
        :return: The fixed range.
        :rtype: tuple
        '''
        return slice(
            min(max(rng.start, 0), data_scales[0].shape[dim]-1)
                if rng.start is not None
                else 0,
            min(max(rng.stop, 1), data_scales[0].shape[dim])
                if rng.stop is not None
                else data_scales[0].shape[dim]
            )

    #########################################
    def _prepare_featurise_voxels(self, data_scales, indexes, output, output_start_row_index, output_start_col_index):
        '''
        (Protected method) Fix any missing parameters sent to featurise_voxels method.

        :param dict data_scales: A dictionary of different scales of the volume.
        :param list indexes: A list of tuples, each of which consists of (slice, row, column)
            such that data_scales[0][(slice, row, column)] refers to the voxel to featurise.
        :param numpy.ndarray output: The output dataset to contain the features (will be created
            if None). Must be 2D.
        :param int output_start_row_index: The row index in the output dataset to start putting
            feature vectors in.
        :param int output_start_col_index: The column index in the output dataset to start putting
            feature vectors from.
        :return: Tuple with fixed parameters (rows_needed, cols_needed, output).
        :rtype: tuple
        '''
        feature_size = self.get_feature_size()

        if output is None:
            output = np.empty((len(indexes), feature_size), np.float32)
        if len(output.shape) != 2 or output.dtype != np.float32:
            raise ValueError('Output array must be a float32 matrix.')

        rows_needed = len(indexes)
        last_output_row_index = output_start_row_index + rows_needed - 1
        if last_output_row_index >= output.shape[0]:
            raise ValueError('Provided output array does not have enough rows to hold result in expected range (array rows = {}, rows needed = {}, last output row index = {}).'.format(output.shape[0], rows_needed, last_output_row_index))

        cols_needed = feature_size
        last_output_col_index = output_start_col_index + cols_needed - 1
        if last_output_col_index >= output.shape[1]:
            raise ValueError('Provided output array does not have enough columns to hold result in expected range (array columns = {}, columns needed = {}, last output column index = {}).'.format(output.shape[1], cols_needed, last_output_col_index))

        return (rows_needed, cols_needed, output)

    #########################################
    def _prepare_featurise_slice(self, data_scales, slice_range, row_range, col_range, output, output_start_row_index, output_start_col_index):
        '''
        (Protected method) Fix any missing parameters sent to featurise_slice method.

        :param dict data_scales: A dictionary of different scales of the volume.
        :param slice slice_range: The range of slices to featurise.
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

        slc_range = self._fix_range(data_scales, 0, slice_range)
        row_range = self._fix_range(data_scales, 1, row_range)
        col_range = self._fix_range(data_scales, 2, col_range)

        if output is None:
            output = np.empty(((slc_range.stop-slc_range.start)*(row_range.stop-row_range.start)*(col_range.stop-col_range.start), feature_size), np.float32)
        if len(output.shape) != 2 or output.dtype != np.float32:
            raise ValueError('Output array must be a float32 matrix.')

        rows_needed = (slc_range.stop-slc_range.start)*(row_range.stop - row_range.start)*(col_range.stop - col_range.start)
        last_output_row_index = output_start_row_index + rows_needed - 1

        if last_output_row_index >= output.shape[0]:
            raise ValueError('Provided output array does not have enough rows to hold result in expected range (array rows = {}, rows needed = {}, last output row index = {}).'.format(output.shape[0], rows_needed, last_output_row_index))

        cols_needed = feature_size
        last_output_col_index = output_start_col_index + cols_needed - 1
        if last_output_col_index >= output.shape[1]:
            raise ValueError('Provided output array does not have enough columns to hold result in expected range (array columns = {}, columns needed = {}, last output column index = {}).'.format(output.shape[1], cols_needed, last_output_col_index))

        return (rows_needed, cols_needed, slc_range, row_range, col_range, output)

    #########################################
    def featurise_voxels(self, data_scales, indexes, output=None, output_start_row_index=0, output_start_col_index=0, dataset_name=None, features_table=None):
        '''
        Turn a set of voxels into a matrix of feature vectors.

        The matrix will have a row for every voxel and the number of columns is the
        feature size. This function is made with creating a dataset in mind, that is, where the
        feature vectors are to be presented as a list rather than in the same shape of the slice.
        In order to be more memory efficient, the result can be placed in a reserved dataset
        matrix that should contain the features of a number of slices. The position of the current
        slice's features within this larger dataset can controlled using the output_start_row_index
        and output_start_col_index parameters.

        :param dict data_scales: A dictionary of different scales of the volume.
        :param list indexes: A list of tuples, each of which consists of (slice, row, column)
            such that data_scales[0][(slice, row, column)] refers to the voxel to featurise.
        :param numpy.ndarray output: The output dataset to contain the features (will be created
            if None). Must be 2D.
        :param int output_start_row_index: The row index in the output dataset to start putting
            feature vectors in.
        :param int output_start_col_index: The column index in the output dataset to start putting
            feature vectors from.
        :param str dataset_name: A unique name for the set of voxels referred to.
            If None then memoisation will not be used.
        :param FeaturesTable features_table: Lookup table to store computed features.
            If None then memoisation will not be used.
        :return: A reference to output.
        :rtype: numpy.ndarray
        '''
        raise NotImplementedError()

    #########################################
    def featurise_slice(self, data_scales, slice_range, block_rows, block_cols, row_range=slice(None), col_range=slice(None), output=None, output_start_row_index=0, output_start_col_index=0, n_jobs=1):
        '''
        Turn a range of slices from a volume into a matrix of feature vectors.

        The matrix will have a row for every voxel in the slice and the number of columns is the
        feature size. This function is made with creating a dataset in mind, that is, where the
        feature vectors are to be presented as a list rather than in the same shape of the slice.
        In order to be more memory efficient, the result can be placed in a reserved dataset
        matrix that should contain the features of a number of slices. The position of the current
        slice's features within this larger dataset can controlled using the output_start_row_index
        and output_start_col_index parameters.

        :param dict data_scales: A dictionary of different scales of the volume.
        :param slice slice_range: The range of slices in the volume to featurise.
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
    def refresh_parameters(self):
        '''
        Refresh parameter values from the samplers provided.
        '''
        pass

    #########################################
    def set_sampler_values(self, config):
        '''
        Set the values of the samplers provided according to a config.

        :param dict config: The configuration dictionary for the feature parameters.
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
    def get_params(self):
        '''
        Get the featuriser's parameters as nested tuples.

        :return: The parameters.
        :rtype: tuple
        '''
        return tuple()

    #########################################
    def featurise_voxels(self, data_scales, indexes, output=None, output_start_row_index=0, output_start_col_index=0, dataset_name=None, features_table=None):
        '''
        Turn a set of voxels into a matrix of feature vectors.

        See super class for more information.

        :param dict data_scales: As described in the super class.
        :param list indexes: As described in the super class.
        :param numpy.ndarray output: As described in the super class.
        :param int output_start_row_index: As described in the super class.
        :param int output_start_col_index: As described in the super class.
        :param str dataset_name: As described in the super class.
        :param FeaturesTable features_table: As described in the super class.
        :return: As described in the super class.
        :rtype: numpy.ndarray
        '''
        (output_rows_needed, output_cols_needed, output) = self._prepare_featurise_voxels(data_scales, indexes, output, output_start_row_index, output_start_col_index)

        if dataset_name is not None and features_table is not None:
            features = features_table.get_features('voxel', self.get_params(), dataset_name)
            if features is not None:
                output[
                    output_start_row_index:output_start_row_index+output_rows_needed,
                    output_start_col_index:output_start_col_index+output_cols_needed
                    ] = features
                return output

        for (out_row, index) in enumerate(indexes):
            output[
                output_start_row_index + out_row,
                output_start_col_index:output_start_col_index+output_cols_needed
                ] = data_scales[0][index]

        if dataset_name is not None and features_table is not None:
            features_table.add_features(
                'voxel', self.get_params(), dataset_name,
                output[
                    output_start_row_index:output_start_row_index+output_rows_needed,
                    output_start_col_index:output_start_col_index+output_cols_needed
                    ]
                )

        return output

    #########################################
    def featurise_slice(self, data_scales, slice_range, block_rows, block_cols, row_range=slice(None), col_range=slice(None), output=None, output_start_row_index=0, output_start_col_index=0, n_jobs=1):
        '''
        Turn a range of slices from a volume into a matrix of feature vectors.

        See super class for more information.

        :param dict data_scales: As described in the super class.
        :param slice slice_range: As described in the super class.
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
        (output_rows_needed, output_cols_needed, slc_range, row_range, col_range, output) = self._prepare_featurise_slice(data_scales, slice_range, row_range, col_range, output, output_start_row_index, output_start_col_index)

        output[
            output_start_row_index:output_start_row_index+output_rows_needed,
            output_start_col_index:output_start_col_index+output_cols_needed
            ] = np.reshape(data_scales[0][slice_range, row_range, col_range], (-1, 1))
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
    def __init__(self, radius, scale, num_bins, use_gpu=False):
        '''
        Constructor.

        :param radius: The neighbourhood radius or a function that generates it.
        :type radius: int or samplers.Sampler
        :param scale: The scale of the volume from which to extract this neighbourhood or a function that generates it.
        :type scale: int or samplers.Sampler
        :param num_bins: num_bins is the number of bins in the histogram or a function that generates it.
        :type num_bins: int or samplers.Sampler
        :param bool use_gpu: Flag indicating the use of GPU implementation.
        '''
        self.radius = None
        self.scale = None
        self.num_bins = None
        self.use_gpu = use_gpu
        self.radius_sampler = None
        self.scale_sampler = None
        self.num_bins_sampler = None
        if isinstance(radius, samplers.Sampler):
            self.radius_sampler = radius
        else:
            self.radius = radius
        if isinstance(scale, samplers.Sampler):
            self.scale_sampler = scale
        else:
            self.scale = scale
        if isinstance(num_bins, samplers.Sampler):
            self.num_bins_sampler = num_bins
        else:
            self.num_bins = num_bins

    #########################################
    def refresh_parameters(self):
        '''
        Refresh parameter values from the samplers provided.
        '''
        self.radius = self.radius_sampler.get_value()
        self.scale = self.scale_sampler.get_value()
        self.num_bins = self.num_bins_sampler.get_value()

    #########################################
    def set_sampler_values(self, config):
        '''
        Set the values of the samplers provided according to a config.

        :param dict config: The configuration dictionary for the feature parameters.
        '''
        self.radius_sampler.set_value(config['params']['radius'])
        self.scale_sampler.set_value(config['params']['scale'])
        self.num_bins_sampler.set_value(config['params']['num_bins'])

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
        return {
            'type': 'histogram',
            'params': {
                'radius': self.radius,
                'scale': self.scale,
                'num_bins': self.num_bins
                }
            }

    #########################################
    def get_params(self):
        '''
        Get the featuriser's parameters as nested tuples.

        :return: The parameters.
        :rtype: tuple
        '''
        return (self.radius, self.scale, self.num_bins)

    #########################################
    def featurise_voxels(self, data_scales, indexes, output=None, output_start_row_index=0, output_start_col_index=0, dataset_name=None, features_table=None):
        '''
        Turn a set of voxels into a matrix of feature vectors.

        See super class for more information.

        :param dict data_scales: As described in the super class.
        :param list indexes: As described in the super class.
        :param numpy.ndarray output: As described in the super class.
        :param int output_start_row_index: As described in the super class.
        :param int output_start_col_index: As described in the super class.
        :param str dataset_name: As described in the super class.
        :param FeaturesTable features_table: As described in the super class.
        :return: As described in the super class.
        :rtype: numpy.ndarray
        '''
        (output_rows_needed, output_cols_needed, output) = self._prepare_featurise_voxels(data_scales, indexes, output, output_start_row_index, output_start_col_index)

        if dataset_name is not None and features_table is not None:
            features = features_table.get_features('histogram', self.get_params(), dataset_name)
            if features is not None:
                output[
                    output_start_row_index:output_start_row_index+output_rows_needed,
                    output_start_col_index:output_start_col_index+output_cols_needed
                    ] = features
                return output

        for (out_row, index) in enumerate(indexes):
            neighbourhood = regions.get_neighbourhood_array_3d(data_scales[self.scale], index, self.radius, {0,1,2}, scale=self.scale)

            feature_vec = histograms.histogram(neighbourhood, self.num_bins, (0, 2**16))

            output[
                out_row,
                output_start_col_index:output_start_col_index+output_cols_needed
                ] = feature_vec

        if dataset_name is not None and features_table is not None:
            features_table.add_features(
                'histogram', self.get_params(), dataset_name,
                output[
                    output_start_row_index:output_start_row_index+output_rows_needed,
                    output_start_col_index:output_start_col_index+output_cols_needed
                    ]
                )

        return output

    #########################################
    def featurise_slice(self, data_scales, slice_range, block_rows, block_cols, row_range=slice(None), col_range=slice(None), output=None, output_start_row_index=0, output_start_col_index=0, n_jobs=1):
        '''
        Turn a range of slices from a volume into a matrix of feature vectors.

        See super class for more information.

        :param dict data_scales: As described in the super class.
        :param slice slice_range: As described in the super class.
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
        (output_rows_needed, output_cols_needed, slc_range, row_range, col_range, output) = self._prepare_featurise_slice(data_scales, slice_range, row_range, col_range, output, output_start_row_index, output_start_col_index)

        def processor(params, radius, scale, num_bins, full_input_ranges, output_start_row_index, output_start_col_index, use_gpu):
            '''Processor for process_array_in_blocks_slice_range.'''
            [ num_slcs_out, num_rows_out, num_cols_out ] = params[0]['contextless_shape']

            if use_gpu:
                hists = histograms.gpu_apply_histogram_to_all_neighbourhoods_in_slice_3d(
                    params[scale]['block'],
                    params[scale]['contextless_slices_wrt_block'][0],
                    radius,
                    {0,1,2},
                    0, 2**16,
                    num_bins,
                    row_slice=params[scale]['contextless_slices_wrt_block'][1], col_slice=params[scale]['contextless_slices_wrt_block'][2]
                    )
            else:
                hists = histograms.apply_histogram_to_all_neighbourhoods_in_slice_3d(
                    params[scale]['block'],
                    params[scale]['contextless_slices_wrt_block'][0],
                    radius,
                    {0,1,2},
                    0, 2**16,
                    num_bins,
                    row_slice=params[scale]['contextless_slices_wrt_block'][1], col_slice=params[scale]['contextless_slices_wrt_block'][2]
                    )

            grown = downscales.grow_array(
                hists,
                scale,
                [0, 1, 2],
                params[0]['contextless_shape'],
                [s.start%(2**scale) for s in params[0]['contextless_slices_wrt_whole']]
                )

            features = np.reshape(grown, (-1, num_bins)).astype(np.float32)

            num_in_rows = full_input_ranges[0].stop - full_input_ranges[0].start
            num_in_cols = full_input_ranges[1].stop - full_input_ranges[1].start
            slice_size = num_in_rows * num_in_cols
            out_indexes = (
                    [
                        output_start_row_index + slc*slice_size + row*num_in_cols + col
                        for slc in range(params[0]['contextless_slices_wrt_range'][0].start, params[0]['contextless_slices_wrt_range'][0].stop)
                        for row in range(params[0]['contextless_slices_wrt_range'][1].start, params[0]['contextless_slices_wrt_range'][1].stop)
                        for col in range(params[0]['contextless_slices_wrt_range'][2].start, params[0]['contextless_slices_wrt_range'][2].stop)
                    ],
                    slice(output_start_col_index, output_start_col_index+num_bins)
                )
            return (features, out_indexes)

        return arrayprocs.process_array_in_blocks_slice_range(
            data_scales,
            output,
            processor,
            block_shape=(block_rows, block_cols),
            slice_range=slice_range,
            scales=self.get_scales_needed(),
            in_ranges=[row_range, col_range],
            context_size=self.get_context_needed(),
            n_jobs=n_jobs,
            extra_params=(self.radius, self.scale, self.num_bins, (row_range, col_range), output_start_row_index, output_start_col_index, self.use_gpu),
            )


#########################################
class LocalBinaryPatternFeaturiser(Featuriser):
    '''
    A featuriser that collects a histogram of LBP patterns at a plane around each voxel.

    The feature vector of a voxel consists of histograms of squares of a given size and
    orientation and at a given scale centered on the voxel. The number of bins in each histogram
    is 10, as the LBP algorithm used is uniform rotation invarient.
    '''

    #########################################
    def __init__(self, neighbouring_dims, radius, scale):
        '''
        Constructor.

        :param neighbouring_dims: The neighbourhood dimensions to keep or a function that generates it.
        :type neighbouring_dims: set
        :param radius: The neighbourhood radius or a function that generates it.
        :type radius: int or samplers.Sampler
        :param scale: The scale of the volume from which to extract this neighbourhood or a function that generates it.
        :type scale: int or samplers.Sampler
        '''
        if not isinstance(neighbouring_dims, set):
            raise ValueError('neighbouring_dims must be a set.')
        if len(neighbouring_dims) != 2 or not neighbouring_dims < {0,1,2}:
            raise ValueError('neighbouring_dims must be {0,1}, {0,2}, or {1,2}.')

        self.neighbouring_dims = neighbouring_dims
        self.radius = None
        self.scale = None
        self.radius_sampler = None
        self.scale_sampler = None
        if isinstance(radius, samplers.Sampler):
            self.radius_sampler = radius
        else:
            self.radius = radius
        if isinstance(scale, samplers.Sampler):
            self.scale_sampler = scale
        else:
            self.scale = scale


    #########################################
    def refresh_parameters(self):
        '''
        Refresh parameter values from the samplers provided.
        '''
        self.radius = self.radius_sampler.get_value()
        self.scale = self.scale_sampler.get_value()

    #########################################
    def set_sampler_values(self, config):
        '''
        Set the values of the samplers provided according to a config.

        :param dict config: The configuration dictionary for the feature parameters.
        '''
        self.radius_sampler.set_value(config['params']['radius'])
        self.scale_sampler.set_value(config['params']['scale'])

    #########################################
    def get_feature_size(self):
        '''
        Get the number of elements in the feature vector.

        :return: The feature vector size.
        :rtype: int
        '''
        return 10

    #########################################
    def get_context_needed(self):
        '''
        Get the maximum amount of context needed around a voxel to generate a feature vector.

        :return: The context size.
        :rtype: int
        '''
        return self.radius + 1

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
        return {
            'type': 'lbp',
            'params': {
                'orientation': {
                    (0,1): 'right',
                    (0,2): 'flat',
                    (1,2): 'front'
                    }[tuple(sorted(self.neighbouring_dims))],
                'radius': self.radius,
                'scale': self.scale
                }
            }

    #########################################
    def get_params(self):
        '''
        Get the featuriser's parameters as nested tuples.

        :return: The parameters.
        :rtype: tuple
        '''
        return (tuple(sorted(self.neighbouring_dims)), self.radius, self.scale)

    #########################################
    def featurise_voxels(self, data_scales, indexes, output=None, output_start_row_index=0, output_start_col_index=0, dataset_name=None, features_table=None):
        '''
        Turn a set of voxels into a matrix of feature vectors.

        See super class for more information.

        :param dict data_scales: As described in the super class.
        :param list indexes: As described in the super class.
        :param numpy.ndarray output: As described in the super class.
        :param int output_start_row_index: As described in the super class.
        :param int output_start_col_index: As described in the super class.
        :param str dataset_name: As described in the super class.
        :param FeaturesTable features_table: As described in the super class.
        :return: As described in the super class.
        :rtype: numpy.ndarray
        '''
        (output_rows_needed, output_cols_needed, output) = self._prepare_featurise_voxels(data_scales, indexes, output, output_start_row_index, output_start_col_index)

        if dataset_name is not None and features_table is not None:
            features = features_table.get_features('lbp', self.get_params(), dataset_name)
            if features is not None:
                output[
                    output_start_row_index:output_start_row_index+output_rows_needed,
                    output_start_col_index:output_start_col_index+output_cols_needed
                    ] = features
                return output

        for (out_row, index) in enumerate(indexes):
            neighbourhood = regions.get_neighbourhood_array_3d(data_scales[self.scale], index, self.radius + 1, self.neighbouring_dims, scale=self.scale)
            lbp = skimage.feature.local_binary_pattern(neighbourhood, 8, 1, 'uniform')[1:-1,1:-1]
            feature_vec = histograms.histogram(lbp, 10, (0, 10))

            output[
                out_row,
                output_start_col_index:output_start_col_index+output_cols_needed
                ] = feature_vec

        if dataset_name is not None and features_table is not None:
            features_table.add_features(
                'lbp', self.get_params(), dataset_name,
                output[
                    output_start_row_index:output_start_row_index+output_rows_needed,
                    output_start_col_index:output_start_col_index+output_cols_needed
                    ]
                )

        return output

    #########################################
    def featurise_slice(self, data_scales, slice_range, block_rows, block_cols, row_range=slice(None), col_range=slice(None), output=None, output_start_row_index=0, output_start_col_index=0, n_jobs=1):
        '''
        Turn a range of slices from a volume into a matrix of feature vectors.

        See super class for more information.

        :param dict data_scales: As described in the super class.
        :param slice slice_range: As described in the super class.
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
        (output_rows_needed, output_cols_needed, slc_range, row_range, col_range, output) = self._prepare_featurise_slice(data_scales, slice_range, row_range, col_range, output, output_start_row_index, output_start_col_index)

        def processor(params, neighbouring_dims, radius, scale, full_input_ranges, output_start_row_index, output_start_col_index):
            '''Processor for process_array_in_blocks_single_slice.'''
            [ num_slcs_out, num_rows_out, num_cols_out ] = params[0]['contextless_shape']

            lbp_codes = np.empty_like(params[scale]['block'])
            if neighbouring_dims == {1,2}:
                for data_index in range(
                        params[scale]['contextless_slices_wrt_block'][0].start,
                        params[scale]['contextless_slices_wrt_block'][0].stop
                    ):
                    lbp_codes[data_index, :, :] = skimage.feature.local_binary_pattern(params[scale]['block'][data_index,:,:], 8, 1, 'uniform')
            else:
                dim = ({0,1,2} - neighbouring_dims).pop()
                index = [ slice(None), slice(None), slice(None) ]
                for i in range(lbp_codes.shape[dim]):
                    index[dim] = i
                    index_ = tuple(index)
                    lbp_codes[index_] = skimage.feature.local_binary_pattern(params[scale]['block'][index_], 8, 1, 'uniform')

            hists = histograms.apply_histogram_to_all_neighbourhoods_in_slice_3d(
                lbp_codes,
                params[scale]['contextless_slices_wrt_block'][0],
                radius,
                neighbouring_dims,
                0, 10,
                10,
                row_slice=params[scale]['contextless_slices_wrt_block'][1], col_slice=params[scale]['contextless_slices_wrt_block'][2]
                )

            grown = downscales.grow_array(
                hists,
                scale,
                [0, 1, 2],
                params[0]['contextless_shape'],
                [s.start%(2**scale) for s in params[0]['contextless_slices_wrt_whole']]
                )

            features = np.reshape(grown, (-1, 10)).astype(np.float32)

            num_in_rows = full_input_ranges[0].stop - full_input_ranges[0].start
            num_in_cols = full_input_ranges[1].stop - full_input_ranges[1].start
            slice_size = num_in_rows * num_in_cols
            out_indexes = (
                    [
                        output_start_row_index + slc*slice_size + row*num_in_cols + col
                        for slc in range(params[0]['contextless_slices_wrt_range'][0].start, params[0]['contextless_slices_wrt_range'][0].stop)
                        for row in range(params[0]['contextless_slices_wrt_range'][1].start, params[0]['contextless_slices_wrt_range'][1].stop)
                        for col in range(params[0]['contextless_slices_wrt_range'][2].start, params[0]['contextless_slices_wrt_range'][2].stop)
                    ],
                    slice(output_start_col_index, output_start_col_index+10)
                )
            return (features, out_indexes)

        return arrayprocs.process_array_in_blocks_slice_range(
            data_scales,
            output,
            processor,
            block_shape=(block_rows, block_cols),
            slice_range=slice_range,
            scales=self.get_scales_needed(),
            in_ranges=[row_range, col_range],
            context_size=self.get_context_needed(),
            n_jobs=n_jobs,
            extra_params=(self.neighbouring_dims, self.radius, self.scale, (row_range, col_range), output_start_row_index, output_start_col_index),
            )


#########################################
class CompositeFeaturiser(Featuriser):
    '''Combine several featurisers into one with the feature vectors being concatenated.'''

    #########################################
    def __init__(self, featuriser_list):
        '''
        Constructor.

        :param list featuriser_list: List of featuriser objects to combine.
        '''
        self.featuriser_list = featuriser_list

    #########################################
    def refresh_parameters(self):
        '''
        Refresh parameter values from the samplers provided.
        '''
        for featuriser in self.featuriser_list:
            featuriser.refresh_parameters()

    #########################################
    def set_sampler_values(self, config):
        '''
        Set the values of the samplers provided according to a config.

        :param dict config: The configuration dictionary for the feature parameters.
        '''
        for (featuriser, featuriser_config) in zip(self.featuriser_list, config['params']['featuriser_list']):
            featuriser.set_sampler_values(featuriser_config)

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
    def get_params(self):
        '''
        Get the featuriser's parameters as nested tuples.

        :return: The parameters.
        :rtype: tuple
        '''
        return tuple(sub_featuriser.get_params() for sub_featuriser in self.featuriser_list)

    #########################################
    def featurise_voxels(self, data_scales, indexes, output=None, output_start_row_index=0, output_start_col_index=0, dataset_name=None, features_table=None):
        '''
        Turn a set of voxels into a matrix of feature vectors.

        See super class for more information.

        :param dict data_scales: As described in the super class.
        :param list indexes: As described in the super class.
        :param numpy.ndarray output: As described in the super class.
        :param int output_start_row_index: As described in the super class.
        :param int output_start_col_index: As described in the super class.
        :param str dataset_name: As described in the super class.
        :param FeaturesTable features_table: As described in the super class.
        :return: As described in the super class.
        :rtype: numpy.ndarray
        '''
        (output_rows_needed, output_cols_needed, output) = self._prepare_featurise_voxels(data_scales, indexes, output, output_start_row_index, output_start_col_index)

        for featuriser in self.featuriser_list:
            featuriser.featurise_voxels(data_scales, indexes, output, output_start_row_index, output_start_col_index, dataset_name, features_table)
            output_start_col_index += featuriser.get_feature_size()

        return output

    #########################################
    def featurise_slice(self, data_scales, slice_range, block_rows, block_cols, row_range=slice(None), col_range=slice(None), output=None, output_start_row_index=0, output_start_col_index=0, n_jobs=1):
        '''
        Turn a range of slices from a volume into a matrix of feature vectors.

        See super class for more information.

        :param dict data_scales: As described in the super class.
        :param slice slice_range: As described in the super class.
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
        (output_rows_needed, output_cols_needed, slc_range, row_range, col_range, output) = self._prepare_featurise_slice(data_scales, slice_range, row_range, col_range, output, output_start_row_index, output_start_col_index)

        for featuriser in self.featuriser_list:
            featuriser.featurise_slice(data_scales, slice_range, block_rows, block_cols, row_range, col_range, output, output_start_row_index, output_start_col_index, n_jobs)
            output_start_col_index += featuriser.get_feature_size()

        return output
