'''Module for volume voxel/label related functions.'''

import os
import PIL.Image
import h5py
import numpy as np
from asemi_segmenter.lib import files
from asemi_segmenter.lib import images


#Control label indexes.
UNINIT_LABEL = 2**8-1
MULTILABEL = 2**8-2
FIRST_CONTROL_LABEL = MULTILABEL


#########################################
class VolumeData(object):
    '''Struct for volume meta data.'''

    #########################################
    def __init__(self, fullfnames, shape):
        '''
        Create a volume object.

        :param list fullfnames: List of all full file names (with path) of all the slices in
            the volume or subvolume.
        :param tuple shape: The shape of a single 2D slice (numpy shape).
        '''
        self.fullfnames = fullfnames
        self.shape = shape


#########################################
class LabelData(object):
    '''Struct for a single label's meta data.'''

    #########################################
    def __init__(self, fullfnames, shape, name):
        '''
        Create a label object.

        :param list fullfnames: List of all full file names (with path) of all the mask slices in
            the subvolume.
        :param tuple shape: The shape of a single 2D slice (numpy shape).
        :param str name: The name of the label.
        '''
        self.fullfnames = fullfnames
        self.shape = shape
        self.name = name


#########################################
class FullVolume(object):
    '''
    Interface for the preprocessed volume HDF file.

    See user guide for description of the HDF file.
    '''

    #########################################
    def __init__(self, data_fullfname):
        '''
        Constructor.

        :param str data_fullfname: The full file name (with path) to the HDF file if loading one
            or to the file to create if not.
        '''
        self.data_fullfname = data_fullfname
        self.data = None
        if self.data_fullfname is None:
            raise NotImplementedError('Non-file preprocessed data method not implemented.')

    #########################################
    def create(self, config_data, volume_shape):
        '''
        Create/overwrite a new HDF file using the file path given in the constructor.

        :param dict config_data: The configuration specs used to preprocess the volume. See user
            guide for description of the preprocess configuration.
        :param tuple volume_shape: 3-tuple describing the dimensions of the
            full-sized volume (numpy shape).
        '''
        if self.data_fullfname is not None:
            with h5py.File(self.data_fullfname, 'w') as data_f:
                data_f.attrs['config'] = json.dumps(config_data)
                for scale in range(config_data['num_downsamples']+1):
                    new_shape = downscales.predict_new_shape(volume_shape, scale)
                    data_f.create_dataset(
                        'volume/scale_{}'.format(scale),
                        new_shape,
                        dtype=np.uint16,
                        chunks=None
                        )
                    data_f['volume/scale_{}'.format(scale)].attrs['scale'] = scale
                data_f.create_dataset(
                    'hashes',
                    [volume_shape[0], config_data['hash_function']['params']['hash_size']],
                    dtype=np.float32,
                    chunks=None
                    )

    #########################################
    def load(self):
        '''Load an existing HDF file using the file path given in the constructor.'''
        if self.data_fullfname is not None:
            self.data = h5py.File(self.data_fullfname, 'r+')

    #########################################
    def get_config(self):
        '''
        Get the preprocessing configuration stored in the HDF file.

        See user guide for description of the preprocess configuration.

        :return: The preprocessing configuration.
        :rtype: dict
        '''
        if self.data_fullfname is not None:
            return json.loads(self.data.attrs['config'])
        return None

    #########################################
    def get_shape(self):
        '''
        Get the volume shape of the full sized volume.

        :return: The shape.
        :rtype: tuple
        '''
        return self.data['volume/scale_0'].shape

    #########################################
    def get_dtype(self):
        '''
        Get the numpy data type of the volume.

        :return: The data type.
        :rtype: numpy.dtype
        '''
        return self.data['volume/scale_0'].dtype

    #########################################
    def get_hashes_dtype(self):
        '''
        Get the numpy data type of the slice hashes.

        :return: The data type.
        :rtype: numpy.dtype
        '''
        return self.data['hashes'].dtype

    #########################################
    def get_scale_array(self, scale):
        '''
        Get the array of a particular scale of volume (lazy loaded).

        :param int scale: Scale of the volume desired.
        :return: The array.
        :rtype: h5py.Dataset
        '''
        return self.data['volume/scale_{}'.format(scale)]

    #########################################
    def get_scales(self):
        '''
        Get the scales present in the HDF file.

        :return: A set of integer scales.
        :rtype: set
        '''
        return {
            self.data['volume/{}'.format(name)].attrs['scale']
            for name in self.data['volume'].keys()
            }

    #########################################
    def get_scale_arrays(self, scales=None):
        '''
        Get a dictionary of volumes at different scales.

        :param set scales: The integer scales to extract.
        :return: The dictionary of arrays.
        :rtype: dict
        '''
        if scales is None:
            scales = self.get_scales()
        return {
            scale: self.data['volume/scale_{}'.format(scale)]
            for scale in set(scales)
            }

    #########################################
    def get_hashes_array(self):
        '''
        Get the array of slice hashes (lazy loaded).

        :return: The array of slice hashes.
        :rtype: h5py.Dataset
        '''
        return self.data['hashes']

    #########################################
    def close(self):
        '''Close the HDF file (if used and open).'''
        if self.data is not None:
            self.data.close()
            self.data = None


#########################################
def load_labels(labels_data):
    '''
    Load label slices as a flattened index array.

    The result will be a 1D ubyte array with each element representing a pixel and all pixels
    (labelled or not) being included. Each element contains a number, with a different number for
    each label (0-based). The numbers FIRST_CONTROL_LABEL and greater are control labels which are
    not actually labels with the number UNINIT_LABEL being used for any pixel that was not labelled
    and the number MULTILABEL being used for any pixel that was labelled by more than one label
    (which is considered invalid). All slices are loaded consecutively in the same 1D array.

    :param list labels_data: A list of LabelData objects, one for each label.
    :return: A single 1D ubyte array.
    :rtype: numpy.ndarray
    '''
    slice_size = np.prod(labels_data[0].shape).tolist()
    label_fullfnames = {label_data.name: label_data.fullfnames for label_data in labels_data}
    labels = sorted(label_fullfnames.keys())
    if len(labels) != len(labels_data):
        raise ValueError('Some labels were declared more than once ([{}]).'.format(
            ', '.join(
                label
                for (label, freq) in collections.Counter(
                    label_data.name for label_data in labels_data
                    )
                if freq > 1
                )
            ))
    num_slices = len(label_fullfnames[labels[0]])
    subvolume_slice_labels = np.full([slice_size*num_slices], UNINIT_LABEL, np.uint8)

    for i in range(num_slices):
        slice_labels = np.full([slice_size], UNINIT_LABEL, np.uint8)
        for (label_index, label) in enumerate(labels):
            label_img = load_image(label_fullfnames[label][i]).reshape([-1])
            label_flags = label_img > np.min(label_img)

            slice_labels = np.where(
                np.logical_and(slice_labels != UNINIT_LABEL, label_flags),
                MULTILABEL,
                slice_labels
                )
            slice_labels = np.where(
                np.logical_and(slice_labels != MULTILABEL, label_flags),
                np.array(label_index, np.uint8),
                slice_labels
                )
        subvolume_slice_labels[i*slice_size:(i+1)*slice_size] = slice_labels

    labels_found = {
        labels[label_index]
        for label_index in set(np.unique(subvolume_slice_labels).tolist()) - {
            UNINIT_LABEL, MULTILABEL
            }
        }
    if len(labels_found) != len(labels):
        raise ValueError('Labelled slices provided do not cover all labels given ' \
            '(missing=[{}]).'.format(', '.join(sorted(set(labels) - labels_found))))

    return subvolume_slice_labels


#########################################
def load_volume_dir(volume_dir):
    '''
    Load and validate volume meta data.

    Validation consists of checking
    * that the slice images are all greyscale and
    * that the slice images are all of the same shape.

    :param str volume_dir: The path to the volume directory.
    :return: A volume data object.
    :rtype: VolumeData
    '''
    if not files.fexists(volume_dir):
        raise ValueError('Volume directory does not exist.')

    volume_fullfnames = []
    with os.scandir(volume_dir) as it:
        for entry in it:
            if entry.name.startswith('.'):
                continue
            if entry.name.split('.')[-1] not in images.IMAGE_EXTS:
                continue
            if entry.is_file():
                volume_fullfnames.append(os.path.join(volume_dir, entry.name))
    if not volume_fullfnames:
        raise ValueError('Volume directory does not have any images.')
    volume_fullfnames.sort()

    slice_shape = None
    for fullfname in volume_fullfnames:
        with PIL.Image.open(fullfname) as f:
            shape = (f.height, f.width)
            if f.mode[0] not in 'LI':
                raise ValueError('Found volume slice that is not a greyscale image ' \
                    '({}).'.format(fullfname))
        if slice_shape is not None:
            if shape != slice_shape:
                raise ValueError('Found differently shaped volume slices ' \
                    '({} and {}).'.format(
                        volume_fullfnames[0], fullfname
                        ))
        else:
            slice_shape = shape

    return VolumeData(volume_fullfnames, slice_shape)


#########################################
def get_volume_slice_indexes_in_subvolume(volume_hashes, subvolume_hashes):
    '''
    Get the index of the corresponding volume slice for each slice in a subvolume.

    A linear nearest neighbour search using Manhattan distance on the hashes is performed.

    :param numpy.ndarray volume_hashes: The 2D float32 hashes array of the volume (one row for
        each slice).
    :param numpy.ndarray subvolume_hashes: The 2D float32 hashes array of the subvolume (one row
        for each slice).
    :return A list of integer indexes such that index[i] is the volume slice index corresponding to
        subvolume slice i.
    :rtype: list
    '''
    indexes = []
    for i in range(subvolume_hashes.shape[0]):
        volume_index = np.argmin(
            np.sum(np.abs(volume_hashes - subvolume_hashes[i, :]), axis=1)
            ).tolist()
        indexes.append(volume_index)
    return indexes


#########################################
def load_label_dir(label_dir):
    '''
    Load and validate label meta data of a single label directory.

    Validation consists of checking
    * that the slice images are all greyscale and
    * that all of slice images are of the same shape.

    :param str label_dir: The path to the label directory.
    :return: A label data object.
    :rtype: LabelData
    '''
    if not files.fexists(label_dir):
        raise ValueError('Label directory does not exist.')
    
    label_name = os.path.split(label_dir)[1]
    if label_name == '':  #If directory ends with a '/' then label name will be an empty string.
        label_name = os.path.split(label_dir[:-1])[1]

    label_fullfnames = []
    with os.scandir(label_dir) as it:
        for entry in it:
            if entry.name.startswith('.'):
                continue
            if entry.name.split('.')[-1] not in images.IMAGE_EXTS:
                continue
            if entry.is_file():
                label_fullfnames.append(os.path.join(label_dir, entry.name))
    if not label_fullfnames:
        raise ValueError('Label directory does not have any images.')
    label_fullfnames.sort()

    slice_shape = None
    for fullfname in label_fullfnames:
        with PIL.Image.open(fullfname) as f:
            shape = (f.height, f.width)
            if f.mode[0] not in 'LI':
                raise ValueError('Found label slice that is not a greyscale image ' \
                    '({}).'.format(
                        fullfname
                        ))
        if slice_shape is not None:
            if shape != slice_shape:
                raise ValueError('Found differently shaped label slices ' \
                    '({} and {}).'.format(
                        label_fullfnames[0], fullfname
                        ))
        else:
            slice_shape = shape

    return LabelData(label_fullfnames, slice_shape, label_name)