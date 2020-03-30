'''Data loading, saving, and validation functions.'''

import collections
import json
import pickle
import random
import warnings
import subprocess
import tempfile
import os
import h5py
import sklearn
import sklearn.ensemble
import skimage.io
import skimage.transform
import PIL.Image
import numpy as np
from asemi_segmenter.lib import files
from asemi_segmenter.lib import downscales
from asemi_segmenter.lib import featurisers
from asemi_segmenter.lib import hashfunctions


#########################################
#Control label indexes.
UNINIT_LABEL = 2**8-1
MULTILABEL = 2**8-2
FIRST_CONTROL_LABEL = MULTILABEL

#Images.
IMAGE_EXTS = set('tiff tif png jp2 bmp'.split(' '))

#Available options.
AVAILABLE_DOWNSAMPLE_FILTER_TYPES = set('gaussian null'.split(' '))
AVAILABLE_HASH_FUNCTIONS = set('random_indexing'.split(' '))
AVAILABLE_FEATURISERS = set('voxel histogram composite'.split(' '))
AVAILABLE_CLASSIFIERS = set('random_forest'.split(' '))


#########################################
class DataException(Exception):
    '''Data related exception such as validation errors.'''
    pass


#########################################
class CheckpointManager(object):
    '''Checkpoint manager keep track of which stages in a process are complete.'''

    #########################################
    def __init__(self, this_command, checkpoint_fullfname, restart_checkpoint=False):
        '''
        Create a new checkpoint manager.

        :param str this_command: The unique name of the command using the checkpoint (serves as
            a namespace).
        :param checkpoint_fullfname: The full file name (with path) of the checkpoint file. If
            None then the checkpoint state is not persisted.
        :type checkpoint_fullfname: str or None
        :param bool restart_checkpoint: Whether to ignore the saved checkpoint and start over.
        '''
        self.this_command = this_command
        self.checkpoint_fullfname = checkpoint_fullfname
        self.checkpoints_ready = dict()
        if self.checkpoint_fullfname is not None:
            if files.fexists(self.checkpoint_fullfname):
                with open(self.checkpoint_fullfname, 'rb') as f:
                    self.checkpoints_ready = pickle.load(f)
                if restart_checkpoint and this_command in self.checkpoints_ready:
                    self.checkpoints_ready.pop(this_command)
            else:
                with open(self.checkpoint_fullfname, 'wb') as f:
                    pickle.dump(self.checkpoints_ready, f)

    #########################################
    def get_next_to_process(self, this_checkpoint):
        '''
        Get next iteration to process according to checkpoint.

        :param str this_checkpoint: The unique name of the current checkpoint.
        :return The next iteration number.
        :rtype int
        '''
        if (
                self.this_command in self.checkpoints_ready and
                this_checkpoint in self.checkpoints_ready[self.this_command]
            ):
            return self.checkpoints_ready[self.this_command][this_checkpoint]
        return 0

    #########################################
    def apply(self, this_checkpoint):
        '''
        Apply a checkpoint for use in a with block.

        If the checkpoint was previously completed then the context manager will return an
        object that can be raised to skip the with block. If not then it will return None
        and at the end of the block will automatically save that the checkpoint was completed.

        Example
        .. code-block:: python
            checkpoint_manager = Checkpoint('command_name', 'checkpoint/fullfname.pkl')
            with checkpoint_manager.apply('checkpoint_name') as ckpt:
                if ckpt is not None:
                    raise ckpt
                #Do something.
            #Now the checkpoint 'checkpoint_name' has been recorded as completed (or was skipped).

        :param str this_checkpoint: The unique name of the current checkpoint.
        :return A context manager.
        '''

        class SkipCheckpoint(Exception):
            '''Special exception for skipping the checkpoint with block.'''
            pass

        class ContextMgr(object):
            '''Context manager for checkpoints.'''

            def __init__(self, checkpoint_obj):
                self.checkpoint_obj = checkpoint_obj

            def __enter__(self):
                if (
                        self.checkpoint_obj.this_command in \
                        self.checkpoint_obj.checkpoints_ready and
                        this_checkpoint in self.checkpoint_obj.checkpoints_ready[
                            self.checkpoint_obj.this_command
                            ]
                    ):
                    return SkipCheckpoint()
                return None

            def __exit__(self, etype, ex, traceback):
                if etype is SkipCheckpoint:
                    return True
                elif etype is None:
                    if self.checkpoint_obj.checkpoint_fullfname is not None:
                        if (
                                self.checkpoint_obj.this_command not in \
                                self.checkpoint_obj.checkpoints_ready
                            ):
                            self.checkpoint_obj.checkpoints_ready[
                                self.checkpoint_obj.this_command] = dict()
                        if this_checkpoint not in self.checkpoint_obj.checkpoints_ready[
                                self.checkpoint_obj.this_command
                            ]:
                            self.checkpoint_obj.checkpoints_ready[
                                self.checkpoint_obj.this_command
                                ][this_checkpoint] = 0
                        self.checkpoint_obj.checkpoints_ready[
                            self.checkpoint_obj.this_command
                            ][this_checkpoint] += 1
                        with open(self.checkpoint_obj.checkpoint_fullfname, 'wb') as f:
                            pickle.dump(self.checkpoint_obj.checkpoints_ready, f, protocol=2)
                return None
        return ContextMgr(self)


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
            return load_preprocess_config_data(json.loads(self.data.attrs['config']))
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
class TrainingSet(object):
    '''Training set of voxel features to voxel labels.'''

    #########################################
    def __init__(self, data_fullfname):
        '''
        Create a training set object.

        :param data_fullfname: The full file name (with path) to the HDF file if to be used or
            None if training set will be a numpy array kept in memory.
        :type data_fullfname: str or None
        '''
        self.data_fullfname = data_fullfname
        self.data = None

    #########################################
    def create(self, num_items, feature_size):
        '''
        Create the HDF file or numpy array.

        :param int num_items: The number of voxels in the training set.
        :param int feature_size: The number of elements in the feature vectors describing
            the voxels.
        '''
        if self.data_fullfname is not None:
            with h5py.File(self.data_fullfname, 'w') as data_f:
                data_f.create_dataset('labels', [num_items], dtype=np.uint8, chunks=None)
                data_f.create_dataset(
                    'features',
                    [num_items, feature_size],
                    dtype=np.float32,
                    chunks=None
                    )
        else:
            self.data = {
                'labels': np.empty([num_items], dtype=np.uint8),
                'features': np.empty([num_items, feature_size], dtype=np.float32)
                }

    #########################################
    def load(self):
        '''Load the HDF file (if data_fullfname was not None).'''
        if self.data_fullfname is not None:
            self.data = h5py.File(self.data_fullfname, 'r+')

    #########################################
    def get_labels_array(self):
        '''
        Get the labels column of the training set.

        :return: An array of labels.
        :rtype: h5py.Dataset or numpy.ndarray
        '''
        return self.data['labels']

    #########################################
    def get_features_array(self):
        '''
        Get the features column of the training set.

        :return: A 2D array of features.
        :rtype: h5py.Dataset or numpy.ndarray
        '''
        return self.data['features']

    #########################################
    def get_sample(self, max_sample_size_per_label, seed=None):
        '''
        Get a random sample of the training set.

        Sample is always kept in memory and the training set will be balanced among labels
        provided that there are enough of each label (otherwise all the items of a label
        will be returned).

        :param int max_sample_size_per_label: The number of items from each label to
            return in the new training set. If there are less items than this then all the items
            are returned.
        :param int seed: The random number generator seed to use when randomly selecting training
            items.
        :return The sub training set.
        :rtype: TrainingSet
        '''
        label_locations = dict()
        for (i, label) in enumerate(self.data['labels'][:].tolist()):
            if label < FIRST_CONTROL_LABEL:
                if label not in label_locations:
                    label_locations[label] = list()
                label_locations[label].append(i)
        num_labels = len(label_locations)

        for label in range(num_labels):
            r = random.Random(seed)
            r.shuffle(label_locations[label])

        all_locations = [
            location
            for label in range(num_labels)
            for location in label_locations[label][:max_sample_size_per_label]]
        all_locations.sort()
        total_items_samples = len(all_locations)

        new_trainingset = TrainingSet(None)
        new_trainingset.create(total_items_samples, self.data['features'].shape[1])
        new_trainingset.get_labels_array()[:] = self.data['labels'][all_locations]
        new_trainingset.get_features_array()[:] = self.data['features'][all_locations, :]

        return new_trainingset

    #########################################
    def close(self):
        '''Close the HDF file (if used and open).'''
        if self.data is not None:
            self.data.close()
            self.data = None


#########################################
class EvaluationResultsFile(object):
    '''Results file interface for the evaluate command.'''

    #########################################
    def __init__(self, results_fullfname):
        '''
        Create an evaluation results file object.

        :param results_fullfname: The full file name (with path) of the results text file. If
            None then no file will be saved and all inputs are ignored.
        :type results_fullfname: str or None
        '''
        self.results_fullfname = results_fullfname

    #########################################
    def create(self, labels):
        '''
        Create the results text file.

        :param list labels: The list of labels used in the segmenter.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'w', encoding='utf-8') as f:
                print(
                    'slice', *labels, 'featurisation duration (s)', 'prediction duration (s)',
                    sep='\t', file=f
                    )

    #########################################
    def append(self, slice_fullfname, ious, featuriser_duration, classifier_duration):
        '''
        Add a new slice's result to the file.

        :param str slice_fullfname: The full file name of the slice being used for evaluation.
        :param list ious: The list of intersection-over-union scores for each label.
        :param float featuriser_duration: The duration of the featurisation process.
        :param float classifier_duration: The duration of the classification process.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'a', encoding='utf-8') as f:
                print(
                    slice_fullfname,
                    *[('{:.3%}'.format(iou) if iou is not None else '') for iou in ious],
                    '{:.1f}'.format(featuriser_duration),
                    '{:.1f}'.format(classifier_duration),
                    sep='\t', file=f
                    )


#########################################
class TuningResultsFile(object):
    '''Results file interface for the tune command.'''

    #########################################
    def __init__(self, results_fullfname):
        '''
        Create a tune results file object.

        :param results_fullfname: The full file name (with path) of the results text file. If
            None then no file will be saved and all inputs are ignored.
        :type results_fullfname: str or None
        '''
        self.results_fullfname = results_fullfname

    #########################################
    def create(self, labels):
        '''
        Create the results text file.

        :param list labels: The list of labels used in the segmenter.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'w', encoding='utf-8') as f:
                print(
                    'json_config', *['{}_iou'.format(label) for label in labels], 'mean_iou', 'min_iou',
                    sep='\t', file=f
                    )

    #########################################
    def append(self, config, average_ious):
        '''
        Add a new slice's result to the file.

        :param dict config: The configuation dictionary used to produce these results.
        :param list ious: The list of average (over slices) intersection-over-union scores for each label.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'a', encoding='utf-8') as f:
                print(
                    json.dumps(config),
                    *['{:.3%}'.format(iou) for iou in average_ious],
                    '{:.3%}'.format(np.mean(average_ious).tolist()),
                    '{:.3%}'.format(np.min(average_ious).tolist()),
                    sep='\t', file=f
                    )


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
def load_image(image_dir):
    '''
    Load an image file as an array. Converts the array to 16-bit first.

    :param str image_dir: The full file name (with path) to the image file.
    :return: The image array as 16-bit.
    :rtype: numpy.ndarray
    '''
    img_data = None
    if image_dir.endswith('.jp2'):
        with tempfile.TemporaryDirectory(dir='/tmp/') as tmp_dir: #Does not work on Windows!
            subprocess.run(
                [
                    'opj_decompress',
                    '-i', image_dir,
                    '-o', os.path.join(tmp_dir, 'tmp.tif')  #Uncompressed image output for speed.
                    ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
                )
            img_data = skimage.io.imread(os.path.join(tmp_dir, 'tmp.tif'))
    else:
        img_data = skimage.io.imread(image_dir)

    #Convert to 16-bit.
    if img_data.dtype == np.uint32:
        img_data = np.right_shift(img_data, 8).astype(np.uint16)
    elif img_data.dtype == np.uint16:
        pass
    elif img_data.dtype == np.uint8:
        img_data = np.left_shift(img_data.astype(np.uint16), 8)

    return img_data


#########################################
def save_image(image_dir, image_data):
    '''
    Save an image array to a file. Suppresses low contrast warnings.

    :param str image_dir: The full file name (with path) to the new image file.
    :param numpy.ndarray image_data: The image array.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        skimage.io.imsave(image_dir, image_data)


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
        raise DataException('Some labels were declared more than once ([{}]).'.format(
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
        raise DataException('Labelled slices provided do not cover all labels given ' \
            '(missing=[{}]).'.format(', '.join(sorted(set(labels) - labels_found))))

    return subvolume_slice_labels


#########################################
def get_subvolume_slice_label_mask(subvolume_slice_labels):
    '''
    An array mask for filtering out all labels in a 1D array which are control labels.

    Given that a labels array can contain control labels, this function lets you get a boolean
    array for masking out control labels when used as an index.

    Example
    .. code-block:: python
        subvolume_slice_labels = load_labels(labels_data)
        subvolume_slice_features = ...

        mask = get_subvolume_slice_label_mask(subvolume_slice_labels)
        filtered_labels = subvolume_slice_labels[mask]
        filtered_features = subvolume_slice_features[mask]

    :param numpy.ndarray subvolume_slice_labels: A 1D ubyte numpy array with a label for each
        pixel in the subvolume.
    :return: A 1D bool numpy array with True wherever there is an actual label in
        subvolume_slice_labels and False wherever there is a control label
        (>= FIRST_CONTROL_LABEL).
    :rtype: numpy.ndarray
    '''
    return subvolume_slice_labels >= FIRST_CONTROL_LABEL


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
        raise DataException('Volume directory does not exist.')

    volume_fullfnames = []
    with os.scandir(volume_dir) as it:
        for entry in it:
            if entry.name.startswith('.'):
                continue
            if entry.name.split('.')[-1] not in IMAGE_EXTS:
                continue
            if entry.is_file():
                volume_fullfnames.append(os.path.join(volume_dir, entry.name))
    if not volume_fullfnames:
        raise DataException('Volume directory does not have any images.')
    volume_fullfnames.sort()

    slice_shape = None
    for fullfname in volume_fullfnames:
        with PIL.Image.open(fullfname) as f:
            shape = (f.height, f.width)
            if f.mode[0] not in 'LI':
                raise DataException('Found volume slice that is not a greyscale image ' \
                    '({}).'.format(fullfname))
        if slice_shape is not None:
            if shape != slice_shape:
                raise DataException('Found differently shaped volume slices ' \
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
        raise DataException('Label directory does not exist.')

    label_name = os.path.split(label_dir)[1]

    label_fullfnames = []
    with os.scandir(label_dir) as it:
        for entry in it:
            if entry.name.startswith('.'):
                continue
            if entry.name.split('.')[-1] not in IMAGE_EXTS:
                continue
            if entry.is_file():
                label_fullfnames.append(os.path.join(label_dir, entry.name))
    if not label_fullfnames:
        raise DataException('Label directory does not have any images.')
    label_fullfnames.sort()

    slice_shape = None
    for fullfname in label_fullfnames:
        with PIL.Image.open(fullfname) as f:
            shape = (f.height, f.width)
            if f.mode[0] not in 'LI':
                raise DataException('Found label slice that is not a greyscale image ' \
                    '({}).'.format(
                        fullfname
                        ))
        if slice_shape is not None:
            if shape != slice_shape:
                raise DataException('Found differently shaped label slices ' \
                    '({} and {}).'.format(
                        label_fullfnames[0], fullfname
                        ))
        else:
            slice_shape = shape

    return LabelData(label_fullfnames, slice_shape, label_name)


#########################################
def validate_annotation_data(full_volume, subvolume_data, labels_data):
    '''
    Validate the preprocessed data of a full volume, with the meta data of a subvolume and labels.

    Validation consists of checking:
    * that the subvolume and all label slices are of the same shape as those of the full volume,
    * that the number of labels does not exceed the built in limit of FIRST_CONTROL_LABEL-1, and
    * that the number of slices in each label is equal to the number of slices in the subvolume.
    '''
    if subvolume_data.shape != full_volume.get_shape()[1:]:
        raise DataException('Subvolume slice shapes do not match volume slice shapes ' \
            '(volume={}, subvolume={}).'.format(
                full_volume.get_shape(), subvolume_data.shape
                ))

    if len(labels_data) > FIRST_CONTROL_LABEL:
        raise DataException('Labels directory has too many labels ({}). ' \
            'Must be less than or equal to {}.'.format(
                len(labels_data), FIRST_CONTROL_LABEL-1
                ))

    for label_data in labels_data:
        if label_data.shape != full_volume.get_shape()[1:]:
            raise DataException('Label {} slice shapes do not match volume slice shapes ' \
                '(volume={}, label={}).'.format(
                    label_data.name,
                    full_volume.get_shape()[1:],
                    label_data.shape
                    ))
        if len(label_data.fullfnames) != len(subvolume_data.fullfnames):
            raise DataException('Number of label slices ({}) in label {} does not equal number ' \
                'of slices in subvolume ({}).'.format(
                    len(label_data.fullfnames),
                    label_data.name,
                    len(subvolume_data.fullfnames)
                    ))


#########################################
def load_preprocess_config_file(config_fullfname):
    '''
    Load and validate a preprocess configuration JSON file and return it as usable objects.

    :param str config_fullfname: Full file name (with path) to the configuration file. See user
        guide for description of the preprocess configuration.
    :return: A tuple containing the following elements in order:
        * The dictionary configuration data.
        * The number of downsamples to apply to the volume (downscale to half the size it was).
        * The loaded downsample filter object (downscales.DownsampleKernel) mentioned in the
            configuration.
        * The loaded hash function object (hashfunctions.HashFunction) mentioned in the
            configuration.
    :rtype: tuple
    '''
    with open(config_fullfname, 'r', encoding='utf-8') as f:
        raw_config = json.load(f)
    return load_preprocess_config_data(raw_config)


#########################################
def load_preprocess_config_data(config_data):
    '''
    Load and validate a preprocess configuration dictionary and return it as usable objects.

    :param dict config_data: The preprocess configuration dictionary. See user guide
        for description of the preprocess configuration.
    :return: A tuple containing the following elements in order:
        * The dictionary configuration data.
        * The number of downsamples to apply to the volume (downscale to half the size it was).
        * The loaded downsample filter object (downscales.DownsampleKernel) mentioned in the
            configuration.
        * The loaded hash function object (hashfunctions.HashFunction) mentioned in the
            configuration.
    :rtype: tuple
    '''
    if not isinstance(config_data, dict):
        raise DataException('Configuration is invalid as it is not in dictionary format.')
    if set(config_data.keys()) != {'num_downsamples', 'downsample_filter', 'hash_function'}:
        raise DataException(
            'Configuration is invalid as it does not have the expected key values.'
            )
    if True:  # pylint: disable=using-constant-test
        if not isinstance(config_data['num_downsamples'], int):
            raise DataException('Configuration is invalid num_downsamples is not an integer.')
        if config_data['num_downsamples'] < 0:
            raise DataException('Configuration is invalid as num_downsamples is negative.')

        if not isinstance(config_data['downsample_filter'], dict):
            raise DataException(
                'Configuration is invalid as downsample_filter is not in dictionary format.'
                )
        if set(config_data['downsample_filter'].keys()) != {'type', 'params'}:
            raise DataException(
                'Configuration is invalid as downsample_filter does not have the expected ' \
                'key values.'
                )
        if True:  # pylint: disable=using-constant-test
            if not isinstance(config_data['downsample_filter']['type'], str):
                raise DataException(
                    'Configuration is invalid as downsample_filter type is not a string.'
                    )
            if config_data['downsample_filter']['type'] not in AVAILABLE_DOWNSAMPLE_FILTER_TYPES:
                raise DataException(
                    'Configuration is invalid as it declares an unexpected downsample_filter type.'
                    )

            if not isinstance(config_data['downsample_filter']['params'], dict):
                raise DataException(
                    'Configuration is invalid as downsample_filter params is not in dictionary ' \
                    'format.'
                    )
            if config_data['downsample_filter']['type'] == 'gaussian':
                if set(config_data['downsample_filter']['params'].keys()) != {'sigma'}:
                    raise DataException(
                        'Configuration is invalid as downsample_filter params does not ' \
                        'have the expected key values for a downsample_filter type ' \
                        'of {}.'.format(config_data['downsample_filter']['type'])
                        )
                if not isinstance(config_data['downsample_filter']['params']['sigma'], float):
                    raise DataException(
                        'Configuration is invalid as downsample_filter params sigma is ' \
                        'not a floating point number.'
                        )
            elif config_data['downsample_filter']['type'] == 'null':
                pass
            else:
                NotImplementedError('Downsample filter {} is not implemented.'.format(
                    config_data['downsample_filter']['type']
                    ))

        if not isinstance(config_data['hash_function'], dict):
            raise DataException(
                'Configuration is invalid as hash_function is not in dictionary format.'
                )
        if set(config_data['hash_function'].keys()) != {'type', 'params'}:
            raise DataException(
                'Configuration is invalid as hash_function does not have the expected key values.'
                )
        if True:  # pylint: disable=using-constant-test
            if not isinstance(config_data['hash_function']['type'], str):
                raise DataException(
                    'Configuration is invalid as hash_function type is not a string.'
                    )
            if config_data['hash_function']['type'] not in AVAILABLE_HASH_FUNCTIONS:
                raise DataException(
                    'Configuration is invalid as it declares an unexpected hash_function type.'
                    )

            if not isinstance(config_data['hash_function']['params'], dict):
                raise DataException(
                    'Configuration is invalid as hash_function params is not in dictionary format.'
                    )
            if config_data['hash_function']['type'] == 'random_indexing':
                if set(config_data['hash_function']['params'].keys()) != {'hash_size'}:
                    raise DataException(
                        'Configuration is invalid as hash_function params does ' \
                        'not have the expected key values for a hash_function type of {}.'.format(
                            config_data['hash_function']['type']
                            )
                        )

                if not isinstance(config_data['hash_function']['params']['hash_size'], int):
                    raise DataException(
                        'Configuration is invalid as hash_function params hash_size is not ' \
                        'an integer.'
                        )
                if config_data['hash_function']['params']['hash_size'] <= 0:
                    raise DataException(
                        'Configuration is invalid as hash_function params hash_size is not a ' \
                        'positive integer.'
                        )
            else:
                NotImplementedError(
                    'Hash function {} is not implemented.'.format(
                        config_data['hash_function']['type']
                        )
                    )

    downsample_filter = {
        'gaussian': downscales.GaussianDownsampleKernel(
            **config_data['downsample_filter']['params']
            ),
        'null': downscales.NullDownsampleKernel(),
        }[
            config_data['downsample_filter']['type']
            ]

    hash_function = {
        'random_indexing': hashfunctions.RandomIndexingHashFunction(
            **config_data['hash_function']['params']
            ),
        }[
            config_data['hash_function']['type']
            ]

    return (config_data, config_data['num_downsamples'], downsample_filter, hash_function)


#########################################
def load_train_config_file(config_fullfname, full_volume=None):
    '''
    Load and validate a train configuration JSON file and return it as usable objects.

    :param str config_fullfname: Full file name (with path) to the configuration file. See user
        guide for description of the train configuration.
    :param full_volume: The full volume object on which to train in order to check that the
        configuration is compatible with it. If None then this check will be skipped.
    :type full_volume: FullVolume or None
    :return: A tuple containing the following elements in order:
        * The dictionary configuration data.
        * The loaded feauteriser object (featurisers.Featuriser) mentioned in the configuration.
        * The loaded classifier object (sklearn classifier) configuration.
    :rtype: tuple
    '''
    with open(config_fullfname, 'r', encoding='utf-8') as f:
        raw_config = json.load(f)
    return load_train_config_data(raw_config, full_volume)


#########################################
def load_train_config_data(config_data, full_volume=None):
    '''
    Load and validate a train configuration dictionary and return it as usable objects.

    :param dict config_data: The train configuration dictionary. See user guide for description
        of the train configuration.
    :param full_volume: The full volume object on which to train in order to check that the
        configuration is compatible with it. If None then this check will be skipped.
    :type full_volume: FullVolume or None
    :return: A tuple containing the following elements in order:
        * The dictionary configuration data.
        * The loaded feauteriser object (featurisers.Featuriser) mentioned in the configuration.
        * The loaded classifier object (sklearn classifier) configuration.
    :rtype: tuple
    '''
    if not isinstance(config_data, dict):
        raise DataException('Configuration is invalid as it is not in dictionary format.')
    if set(config_data.keys()) != {'featuriser', 'classifier', 'training_set'}:
        raise DataException(
            'Configuration is invalid as it does not have the expected key values.'
            )

    if True:  # pylint: disable=using-constant-test
        
        def get_featuriser(featuriser_config):
            '''Create a featuriser object from configuration (recurive).'''
            if not isinstance(featuriser_config, dict):
                raise DataException('Configuration is invalid as featuriser is not in dictionary format.')
            if set(featuriser_config.keys()) != {'type', 'params'}:
                raise DataException(
                    'Configuration is invalid as featuriser does not have the expected key values.'
                    )
            
            if not isinstance(featuriser_config['type'], str):
                raise DataException('Configuration is invalid as featuriser type is not a string.')
            if featuriser_config['type'] not in AVAILABLE_FEATURISERS:
                raise DataException(
                    'Configuration is invalid as it declares an unexpected featuriser type.'
                    )

            if not isinstance(featuriser_config['params'], dict):
                raise DataException(
                    'Configuration is invalid as featuriser params is not in dictionary format.'
                    )
            
            if featuriser_config['type'] == 'voxel':
                if (
                        set(featuriser_config['params'].keys()) != set()
                    ):
                    raise DataException(
                        'Configuration is invalid as featuriser params does not have the ' \
                        'expected key values for a featuriser type of {}.'.format(
                            featuriser_config['type']
                            )
                        )
                return featurisers.VoxelFeaturiser()
                
            elif featuriser_config['type'] == 'histogram':
                if (
                        set(featuriser_config['params'].keys()) != \
                        {'radius', 'scale', 'num_bins'}
                    ):
                    raise DataException(
                        'Configuration is invalid as featuriser params does not have the ' \
                        'expected key values for a featuriser type of {}.'.format(
                            featuriser_config['type']
                            )
                        )

                if not isinstance(featuriser_config['params']['radius'], int):
                    raise DataException(
                        'Configuration is invalid as featuriser params radius ' \
                        'is not an integer.'
                        )
                if not isinstance(featuriser_config['params']['scale'], int):
                    raise DataException(
                        'Configuration is invalid as featuriser params scale is ' \
                        'not an integer.'
                        )
                if not isinstance(featuriser_config['params']['num_bins'], int):
                    raise DataException(
                        'Configuration is invalid as featuriser params num_bins ' \
                        'is not an integer.'
                        )

                return featurisers.HistogramFeaturiser(
                    featuriser_config['params']['radius'],
                    featuriser_config['params']['scale'],
                    featuriser_config['params']['num_bins']
                    )
            
            elif featuriser_config['type'] == 'composite':
                if (set(featuriser_config['params'].keys()) != {'featuriser_list'}):
                    raise DataException(
                        'Configuration is invalid as featuriser params does not have the ' \
                        'expected key values for a featuriser type of {}.'.format(
                            featuriser_config['type']
                            )
                        )
                
                if not isinstance(featuriser_config['params']['featuriser_list'], list):
                    raise DataException(
                        'Configuration is invalid as featuriser params featuriser_list ' \
                        'is not a list.'
                        )
                
                return featurisers.CompositeFeaturiser(
                    [get_featuriser(sub_featuriser_config) for sub_featuriser_config in featuriser_config['params']['featuriser_list']]
                    )
            
            else:
                raise NotImplementedError(
                    'Featuriser {} is not implemented.'.format(featuriser_config['type'])
                    )
        
        featuriser = get_featuriser(config_data['featuriser'])
        
    if not isinstance(config_data['classifier'], dict):
        raise DataException('Configuration is invalid as classifier is not in dictionary format.')
    if set(config_data['classifier'].keys()) != {'type', 'params'}:
        raise DataException(
            'Configuration is invalid as classifier does not have the expected key values.'
            )
    if True:  # pylint: disable=using-constant-test
        if not isinstance(config_data['classifier']['type'], str):
            raise DataException('Configuration is invalid as classifier type is not a string.')
        if config_data['classifier']['type'] not in AVAILABLE_CLASSIFIERS:
            raise DataException(
                'Configuration is invalid as it declares an unexpected classifier type.'
                )

        if not isinstance(config_data['classifier']['params'], dict):
            raise DataException(
                'Configuration is invalid as classifier params is not in dictionary format.'
                )
        if config_data['classifier']['type'] == 'random_forest':
            if (
                    set(config_data['classifier']['params'].keys()) != \
                    {'n_estimators', 'max_depth', 'min_samples_leaf'}
                ):
                raise DataException(
                    'Configuration is invalid as classifier params does not have the expected ' \
                    'key values for a classifier type of {}.'.format(
                        config_data['classifier']['type']
                        )
                    )

            if not isinstance(config_data['classifier']['params']['n_estimators'], int):
                raise DataException(
                    'Configuration is invalid as classifier params n_estimators is not an integer.'
                    )

            if not isinstance(config_data['classifier']['params']['max_depth'], int):
                raise DataException(
                    'Configuration is invalid as classifier params max_depth is not an integer.'
                    )

            if not isinstance(config_data['classifier']['params']['min_samples_leaf'], int):
                raise DataException(
                    'Configuration is invalid as classifier params min_samples_leaf is not an ' \
                    'integer.'
                    )
        else:
            raise NotImplementedError(
                'Classifier {} is not implemented.'.format(
                    config_data['classifier']['type']
                    )
                )

    if not isinstance(config_data['training_set'], dict):
        raise DataException(
            'Configuration is invalid as training_set is not in dictionary format.'
            )
    if set(config_data['training_set'].keys()) != {'sample_size_per_label'}:
        raise DataException(
            'Configuration is invalid as training_set does not have the expected key values.'
            )
    if True:  # pylint: disable=using-constant-test
        if not isinstance(config_data['training_set']['sample_size_per_label'], int):
            raise DataException(
                'Configuration is invalid as training_set sample_size_per_label is not an integer.'
                )
        if (
                config_data['training_set']['sample_size_per_label'] == 0 or
                config_data['training_set']['sample_size_per_label'] < -1
            ):
            raise DataException(
                'Configuration is invalid as training_set sample_size_per_label is 0 or a ' \
                'negative number other than -1.'
                )

    classifier = {
        'random_forest': sklearn.ensemble.RandomForestClassifier(
            **config_data['classifier']['params'],
            random_state=0
            ),
        }[
            config_data['classifier']['type']
            ]

    if full_volume is not None:
        #If scales needed is not a subset of scales available:
        if not featuriser.get_scales_needed() <= full_volume.get_scales():
            raise DataException(
                'Featuriser requires scales that are not included in preprocessed volume ' \
                '(missing scales=[{}]).'.format(
                    ', '.join(sorted(featuriser.get_scales_needed() - full_volume.get_scales()))
                    )
                )

    return (config_data, featuriser, classifier)


#########################################
def load_tune_config_file(config_fullfname):
    '''
    Load and validate a tune configuration JSON file and return it as usable objects.

    :param str config_fullfname: Full file name (with path) to the configuration file. See user
        guide for description of the tune configuration.
    :return: A tuple containing the following elements in order:
        * The dictionary configuration data.
        * The loaded feauteriser object (featurisers.Featuriser) mentioned in the configuration.
        * The loaded classifier object (sklearn classifier) configuration.
    :rtype: tuple
    '''
    with open(config_fullfname, 'r', encoding='utf-8') as f:
        raw_config = json.load(f)
    return load_tune_config_data(raw_config)


#########################################
def load_tune_config_data(config_data):
    '''
    Load and validate a tune configuration dictionary and return it as usable objects.

    :param dict config_data: The tune configuration dictionary. See user guide for description
        of the tune configuration.
    :return: A tuple containing the following elements in order:
        * The dictionary configuration data.
        * The loaded feauteriser object (featurisers.Featuriser) mentioned in the configuration.
        * The loaded classifier object (sklearn classifier) configuration.
    :rtype: tuple
    '''
    rand = random.Random(0)
    
    if not isinstance(config_data, dict):
        raise DataException('Configuration is invalid as it is not in dictionary format.')
    if set(config_data.keys()) != {'featuriser', 'classifier', 'training_set', 'tuning'}:
        raise DataException(
            'Configuration is invalid as it does not have the expected key values.'
            )

    if True:  # pylint: disable=using-constant-test
        
        def get_featuriser(featuriser_config):
            '''Create a featuriser object from configuration (recurive).'''
            if not isinstance(featuriser_config, dict):
                raise DataException('Configuration is invalid as featuriser is not in dictionary format.')
            if set(featuriser_config.keys()) != {'type', 'params'}:
                raise DataException(
                    'Configuration is invalid as featuriser does not have the expected key values.'
                    )
            
            if not isinstance(featuriser_config['type'], str):
                raise DataException('Configuration is invalid as featuriser type is not a string.')
            if featuriser_config['type'] not in AVAILABLE_FEATURISERS:
                raise DataException(
                    'Configuration is invalid as it declares an unexpected featuriser type.'
                    )

            if not isinstance(featuriser_config['params'], dict):
                raise DataException(
                    'Configuration is invalid as featuriser params is not in dictionary format.'
                    )
            
            if featuriser_config['type'] == 'voxel':
                if (
                        set(featuriser_config['params'].keys()) != set()
                    ):
                    raise DataException(
                        'Configuration is invalid as featuriser params does not have the ' \
                        'expected key values for a featuriser type of {}.'.format(
                            featuriser_config['type']
                            )
                        )
                return featurisers.VoxelFeaturiser()
                
            elif featuriser_config['type'] == 'histogram':
                if (
                        set(featuriser_config['params'].keys()) != \
                        {'radius', 'scale', 'num_bins'}
                    ):
                    raise DataException(
                        'Configuration is invalid as featuriser params does not have the ' \
                        'expected key values for a featuriser type of {}.'.format(
                            featuriser_config['type']
                            )
                        )

                radius = None
                if isinstance(featuriser_config['params']['radius'], int):
                    radius = featuriser_config['params']['radius']
                elif isinstance(featuriser_config['params']['radius'], dict):
                    if set(featuriser_config['params']['radius'].keys()) != {'max', 'min'}:
                        raise DataException(
                            'Configuration is invalid as featuriser params radius does not ' \
                            'have the expected key values.'
                            )
                    if not isinstance(featuriser_config['params']['radius']['max'], int):
                        raise DataException(
                            'Configuration is invalid as featuriser params radius max' \
                            'is not an integer.'
                            )
                    if not isinstance(featuriser_config['params']['radius']['min'], int):
                        raise DataException(
                            'Configuration is invalid as featuriser params radius min' \
                            'is not an integer.'
                            )
                    if (
                            featuriser_config['params']['radius']['min'] >= \
                            featuriser_config['params']['radius']['max']
                        ):
                        raise DataException(
                            'Configuration is invalid as featuriser params radius min' \
                            'is not less than featuriser params radius max.'
                            )
                    radius = lambda:rand.randrange(
                        featuriser_config['params']['radius']['min'],
                        featuriser_config['params']['radius']['max'] + 1
                        )
                
                scale = None
                if isinstance(featuriser_config['params']['scale'], int):
                    scale = featuriser_config['params']['scale']
                elif isinstance(featuriser_config['params']['scale'], dict):
                    if set(featuriser_config['params']['scale'].keys()) != {'max', 'min'}:
                        raise DataException(
                            'Configuration is invalid as featuriser params scale does not ' \
                            'have the expected key values.'
                            )
                    if not isinstance(featuriser_config['params']['scale']['max'], int):
                        raise DataException(
                            'Configuration is invalid as featuriser params scale max' \
                            'is not an integer.'
                            )
                    if not isinstance(featuriser_config['params']['scale']['min'], int):
                        raise DataException(
                            'Configuration is invalid as featuriser params scale min' \
                            'is not an integer.'
                            )
                    if (
                            featuriser_config['params']['scale']['min'] >= \
                            featuriser_config['params']['scale']['max']
                        ):
                        raise DataException(
                            'Configuration is invalid as featuriser params scale min' \
                            'is not less than featuriser params scale max.'
                            )
                    scale = lambda:rand.randrange(
                        featuriser_config['params']['scale']['min'],
                        featuriser_config['params']['scale']['max'] + 1
                        )
                
                num_bins = None
                if isinstance(featuriser_config['params']['num_bins'], int):
                    num_bins = featuriser_config['params']['num_bins']
                elif isinstance(featuriser_config['params']['num_bins'], dict):
                    if set(featuriser_config['params']['num_bins'].keys()) != {'max', 'min'}:
                        raise DataException(
                            'Configuration is invalid as featuriser params num_bins does not ' \
                            'have the expected key values.'
                            )
                    if not isinstance(featuriser_config['params']['num_bins']['max'], int):
                        raise DataException(
                            'Configuration is invalid as featuriser params num_bins max' \
                            'is not an integer.'
                            )
                    if not isinstance(featuriser_config['params']['num_bins']['min'], int):
                        raise DataException(
                            'Configuration is invalid as featuriser params num_bins min' \
                            'is not an integer.'
                            )
                    if (
                            featuriser_config['params']['num_bins']['min'] >= \
                            featuriser_config['params']['num_bins']['max']
                        ):
                        raise DataException(
                            'Configuration is invalid as featuriser params num_bins min' \
                            'is not less than featuriser params num_bins max.'
                            )
                    num_bins = lambda:rand.randrange(
                        featuriser_config['params']['num_bins']['min'],
                        featuriser_config['params']['num_bins']['max'] + 1
                        )

                return featurisers.HistogramFeaturiser(radius, scale, num_bins)
            
            elif featuriser_config['type'] == 'composite':
                if (set(featuriser_config['params'].keys()) != {'featuriser_list'}):
                    raise DataException(
                        'Configuration is invalid as featuriser params does not have the ' \
                        'expected key values for a featuriser type of {}.'.format(
                            featuriser_config['type']
                            )
                        )
                
                if not isinstance(featuriser_config['params']['featuriser_list'], list):
                    raise DataException(
                        'Configuration is invalid as featuriser params featuriser_list ' \
                        'is not a list.'
                        )
                
                return featurisers.CompositeFeaturiser(
                    [get_featuriser(sub_featuriser_config) for sub_featuriser_config in featuriser_config['params']['featuriser_list']]
                    )
            
            else:
                raise NotImplementedError(
                    'Featuriser {} is not implemented.'.format(featuriser_config['type'])
                    )
        
        featuriser = get_featuriser(config_data['featuriser'])
        
    if not isinstance(config_data['classifier'], dict):
        raise DataException('Configuration is invalid as classifier is not in dictionary format.')
    if set(config_data['classifier'].keys()) != {'type', 'params'}:
        raise DataException(
            'Configuration is invalid as classifier does not have the expected key values.'
            )
    if True:  # pylint: disable=using-constant-test
        if not isinstance(config_data['classifier']['type'], str):
            raise DataException('Configuration is invalid as classifier type is not a string.')
        if config_data['classifier']['type'] not in AVAILABLE_CLASSIFIERS:
            raise DataException(
                'Configuration is invalid as it declares an unexpected classifier type.'
                )

        if not isinstance(config_data['classifier']['params'], dict):
            raise DataException(
                'Configuration is invalid as classifier params is not in dictionary format.'
                )
        if config_data['classifier']['type'] == 'random_forest':
            if (
                    set(config_data['classifier']['params'].keys()) != \
                    {'n_estimators', 'max_depth', 'min_samples_leaf'}
                ):
                raise DataException(
                    'Configuration is invalid as classifier params does not have the expected ' \
                    'key values for a classifier type of {}.'.format(
                        config_data['classifier']['type']
                        )
                    )

            if not isinstance(config_data['classifier']['params']['n_estimators'], int):
                raise DataException(
                    'Configuration is invalid as classifier params n_estimators is not an integer.'
                    )

            if not isinstance(config_data['classifier']['params']['max_depth'], int):
                raise DataException(
                    'Configuration is invalid as classifier params max_depth is not an integer.'
                    )

            if not isinstance(config_data['classifier']['params']['min_samples_leaf'], int):
                raise DataException(
                    'Configuration is invalid as classifier params min_samples_leaf is not an ' \
                    'integer.'
                    )
        else:
            raise NotImplementedError(
                'Classifier {} is not implemented.'.format(
                    config_data['classifier']['type']
                    )
                )

    if not isinstance(config_data['training_set'], dict):
        raise DataException(
            'Configuration is invalid as training_set is not in dictionary format.'
            )
    if set(config_data['training_set'].keys()) != {'sample_size_per_label'}:
        raise DataException(
            'Configuration is invalid as training_set does not have the expected key values.'
            )
    if True:  # pylint: disable=using-constant-test
        if not isinstance(config_data['training_set']['sample_size_per_label'], int):
            raise DataException(
                'Configuration is invalid as training_set sample_size_per_label is not an integer.'
                )
        if (
                config_data['training_set']['sample_size_per_label'] == 0 or
                config_data['training_set']['sample_size_per_label'] < -1
            ):
            raise DataException(
                'Configuration is invalid as training_set sample_size_per_label is 0 or a ' \
                'negative number other than -1.'
                )
    
    if not isinstance(config_data['tuning'], dict):
        raise DataException(
            'Configuration is invalid as tuning is not in dictionary format.'
            )
    if set(config_data['tuning'].keys()) != {'num_iterations'}:
        raise DataException(
            'Configuration is invalid as tuning does not have the expected key values.'
            )
    if True:  # pylint: disable=using-constant-test
        if not isinstance(config_data['tuning']['num_iterations'], int):
            raise DataException(
                'Configuration is invalid as tuning num_iterations is not an integer.'
                )
        if config_data['tuning']['num_iterations'] <= 0:
            raise DataException(
                'Configuration is invalid as tuning num_iterations is not a positive integer.'
                )

    classifier = {
        'random_forest': sklearn.ensemble.RandomForestClassifier(
            **config_data['classifier']['params'],
            random_state=0
            ),
        }[
            config_data['classifier']['type']
            ]

    return (config_data, featuriser, classifier)
    
    
#########################################
def load_segment_config_file(config_fullfname):
    '''
    Load and validate a segment configuration JSON file.

    :param str config_fullfname: Full file name (with path) to the configuration file. See user
        guide for description of the segment configuration.
    :return: A tuple containing the following elements in order:
        * The dictionary configuration data.
        * Whether to perform soft segmentation or not.
    :rtype: tuple
    '''
    with open(config_fullfname, 'r', encoding='utf-8') as f:
        raw_config = json.load(f)
    return load_segment_config_data(raw_config)


#########################################
def load_segment_config_data(config_data):
    '''
    Load and validate a segment configuration dictionary.

    :param dict config_data: The segment configuration dictionary. See user guide
        for description of the segment configuration.
    :return: A tuple containing the following elements in order:
        * The dictionary configuration data.
        * Whether to perform soft segmentation (greyscale mask) or not (binary mask).
    :rtype: tuple
    '''
    if not isinstance(config_data, dict):
        raise DataException('Configuration is invalid as it is not in dictionary format.')
    if set(config_data.keys()) != {'soft_segmentation'}:
        raise DataException(
            'Configuration is invalid as it does not have the expected key values.'
            )
    if True:  # pylint: disable=using-constant-test
        if not isinstance(config_data['soft_segmentation'], str):
            raise DataException('Configuration is invalid soft_segmentation is not a string.')
        if config_data['soft_segmentation'] not in { 'yes', 'no' }:
            raise DataException('Configuration is invalid as soft_segmentation is not "yes" or "no".')

    return (config_data, config_data['soft_segmentation'] == 'yes')


#########################################
def load_model_file(model_fullfname, full_volume=None):
    '''
    Load and validate a classifier model pickle file and return it as usable objects.

    :param str model_fullfname: Full file name (with path) to the model file. See user
        guide for description of the model data.
    :param full_volume: The full volume object on which to train in order to check that the
        train configuration stored in the model is compatible with it. If None then this check
        will be skipped.
    :type full_volume: FullVolume or None
    :return: A tuple containing the following elements in order:
        * The dictionary train configuration data that was used to train the model.
        * The list of label names.
        * The loaded feauteriser object (featurisers.Featuriser) mentioned in the configuration.
        * The loaded classifier object (sklearn classifier) configuration.
    :rtype: tuple
    '''
    if not files.fexists(model_fullfname):
        raise DataException('Model does not exist.')

    with open(model_fullfname, 'rb') as f:
        model_data = pickle.load(f)

    return load_model_data(model_data, full_volume)


#########################################
def load_model_data(model_data, full_volume=None):
    '''
    Load and validate a classifier model dictionary and return it as usable objects.

    :param dict model_data: The model dictionary. See user guide for description of the model data.
    :param full_volume: The full volume object on which to train in order to check that the
        train configuration stored in the model is compatible with it. If None then this check
        will be skipped.
    :type full_volume: FullVolume or None
    :return: A tuple containing the following elements in order:
        * The dictionary train configuration data that was used to train the model.
        * The list of label names.
        * The loaded feauteriser object (featurisers.Featuriser) mentioned in the configuration.
        * The loaded classifier object (sklearn classifier) configuration.
    :rtype: tuple
    '''
    if not isinstance(model_data, dict):
        raise DataException('Model is invalid as it is not in dictionary format.')
    if set(model_data.keys()) != {'model', 'labels', 'config'}:
        raise DataException('Model is invalid as it does not have the expected key values.')

    (config_data, featuriser, _) = load_train_config_data(model_data['config'], full_volume)

    if not isinstance(model_data['labels'], list):
        raise DataException('Model is invalid as labels are not a list.')
    for (i, entry) in enumerate(model_data['labels']):
        if not isinstance(entry, str):
            raise DataException('Model is invalid as label entry {} is not a string.'.format(i))

    if config_data['classifier'] == 'random_forest':
        if not isinstance(model_data['model'], sklearn.ensemble.RandomForestClassifier):
            raise DataException('Model is invalid as it is not a random forest as declared.')
    if model_data['model'].n_classes_ != len(model_data['labels']):
        raise DataException(
            'Model is invalid as the number of classes is not as declared (declared={}, ' \
            'actual={}).'.format(
                len(model_data['labels']),
                model_data['model'].n_classes_
                )
            )
    if config_data['featuriser'] == 'histograms':
        if model_data['model'].n_features_ != featuriser.get_feature_size():
            raise DataException(
                'Model is invalid as the number of features is not as declared (declared={}, ' \
                'actual={}).'.format(
                    featuriser.get_feature_size(),
                    model_data['model'].n_features_
                    )
                )

    if full_volume is not None:
        #If scales needed is not a subset of scales available:
        if not featuriser.get_scales_needed() <= full_volume.get_scales():
            raise DataException(
                'Featuriser requires scales that are not included in preprocessed volume ' \
                '(missing scales=[{}]).'.format(
                    ', '.join(sorted(featuriser.get_scales_needed() - full_volume.get_scales()))
                    )
                )

    return (config_data, model_data['labels'], featuriser, model_data['model'])


#########################################
def save_model(model_fullfname, model):
    '''
    Save a training model to a pickle file.

    Pickle uses protocol 2.

    :param str model_fullfname: The full file name (with path) of the pickle file.
    :param dict model: The train model dictionary.
    '''
    if model_fullfname is not None:
        with open(model_fullfname, 'wb') as f:
            pickle.dump(model, f, protocol=2)


#########################################
def check_preprocessed_filename(data_fullfname, must_exist=False):
    '''
    Check that the file name for the preprocessed volume is valid.

    Validation consists of checking:
    * that the directory to the file exists,
    * that the file name ends with .hdf, and
    * that the file exists if must_exist is true.

    :param str data_fullfname: The full file name (with path) to the volume.
    :param bool must_exist: Whether the file should already exist or not.
    '''
    dir_path = os.path.split(data_fullfname)[0]
    if dir_path != '' and not files.fexists(dir_path):
        raise DataException('Preprocessed file\'s directory does not exist.')
    if not data_fullfname.endswith('.hdf'):
        raise DataException('Preprocessed file\'s file name does not end with .hdf.')
    if must_exist and not files.fexists(data_fullfname):
        raise DataException('Preprocessed file does not exist.')


#########################################
def check_checkpoint_filename(checkpoint_fullfname, must_exist=False):
    '''
    Check that the file name for the checkpoint pickle is valid.

    Validation consists of checking:
    * that the directory to the file exists,
    * that the file name ends with .pkl, and
    * that the file exists if must_exist is true.

    :param str checkpoint_fullfname: The full file name (with path) to the pickle.
    :param bool must_exist: Whether the file should already exist or not.
    '''
    dir_path = os.path.split(checkpoint_fullfname)[0]
    if dir_path != '' and not files.fexists(dir_path):
        raise DataException('Checkpoint directory does not exist.')
    if not checkpoint_fullfname.endswith('.pkl'):
        raise DataException('Checkpoint file name does not end with .pkl.')
    if must_exist and not files.fexists(checkpoint_fullfname):
        raise DataException('Checkpoint file does not exist.')


#########################################
def check_model_filename(model_fullfname, must_exist=False):
    '''
    Check that the file name for the model pickle is valid.

    Validation consists of checking:
    * that the directory to the file exists,
    * that the file name ends with .pkl, and
    * that the file exists if must_exist is true.

    :param str model_fullfname: The full file name (with path) to the pickle.
    :param bool must_exist: Whether the file should already exist or not.
    '''
    dir_path = os.path.split(model_fullfname)[0]
    if dir_path != '' and not files.fexists(dir_path):
        raise DataException('Model file\'s directory does not exist.')
    if not model_fullfname.endswith('.pkl'):
        raise DataException('Model file\'s file name does not end with .pkl.')
    if must_exist and not files.fexists(model_fullfname):
        raise DataException('Model file does not exist.')


#########################################
def check_evaluation_results_filename(evaluation_results_fullfname, must_exist=False):
    '''
    Check that the file name for the results text file is valid.

    Validation consists of checking:
    * that the directory to the file exists,
    * that the file name ends with .txt, and
    * that the file exists if must_exist is true.

    :param str evaluation_results_fullfname: The full file name (with path) to the text file.
    :param bool must_exist: Whether the file should already exist or not.
    '''
    dir_path = os.path.split(evaluation_results_fullfname)[0]
    if dir_path != '' and not files.fexists(dir_path):
        raise DataException('Result file\'s directory does not exist.')
    if not evaluation_results_fullfname.endswith('.txt'):
        raise DataException('Result file\'s file name does not end with .txt.')
    if must_exist and not files.fexists(evaluation_results_fullfname):
        raise DataException('Result file does not exist.')


#########################################
def check_tune_results_filename(tune_results_fullfname, must_exist=False):
    '''
    Check that the file name for the results text file is valid.

    Validation consists of checking:
    * that the directory to the file exists,
    * that the file name ends with .txt, and
    * that the file exists if must_exist is true.

    :param str tune_results_fullfname: The full file name (with path) to the text file.
    :param bool must_exist: Whether the file should already exist or not.
    '''
    dir_path = os.path.split(evaluation_results_fullfname)[0]
    if dir_path != '' and not files.fexists(dir_path):
        raise DataException('Result file\'s directory does not exist.')
    if not evaluation_results_fullfname.endswith('.txt'):
        raise DataException('Result file\'s file name does not end with .txt.')
    if must_exist and not files.fexists(evaluation_results_fullfname):
        raise DataException('Result file does not exist.')


#########################################
def check_segmentation_results_directory(results_dir):
    '''
    Check that the directory for the segmentation result is valid.

    Validation consists of checking:
    * that the directory.

    :param str results_dir: The path to the directory.
    '''
    if not files.fexists(results_dir):
        raise DataException('Results directory does not exist.')
