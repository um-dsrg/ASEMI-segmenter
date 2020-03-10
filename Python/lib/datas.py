import skimage.io
import skimage.transform
import PIL.Image
import sklearn
import sklearn.ensemble
import numpy as np
import collections
import json
import pickle
import h5py
import random
import warnings
import subprocess
import tempfile
import os
import sys
sys.path.append(os.path.join('..', 'lib'))
import files
import downscales
import featurisers
import hashfunctions

#########################################
UNINIT_LABEL = 2**8-1
MULTILABEL = 2**8-2
first_control_label = MULTILABEL

image_exts = '.tiff .tif .png .jp2 .bmp'.split(' ')
available_downsample_filter_types = set('gaussian null'.split(' '))
available_hash_functions = set('random_indexing'.split(' '))
available_featurisers = set('histograms'.split(' '))
available_classifiers = set('random_forest'.split(' '))

#########################################
class DataException(Exception):
    pass

#########################################
class Checkpoint(object):
    
    #########################################
    def __init__(self, this_command, checkpoint_fullfname, restart_checkpoint=False):
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
        if (
                self.this_command in self.checkpoints_ready and
                this_checkpoint in self.checkpoints_ready[self.this_command]
            ):
            return self.checkpoints_ready[self.this_command][this_checkpoint]
        else:
            return 0
    
    #########################################
    def apply(self, this_checkpoint):
        class SkipCheckpoint(Exception):
            pass
        class Context(object):
            def __init__(self, checkpoint_obj):
                self.checkpoint_obj = checkpoint_obj
            def __enter__(self):
                if (
                        self.checkpoint_obj.this_command in self.checkpoint_obj.checkpoints_ready and
                        this_checkpoint in self.checkpoint_obj.checkpoints_ready[self.checkpoint_obj.this_command]
                    ):
                    return SkipCheckpoint()
                else:
                    return None
            def __exit__(self, etype, e, traceback):
                if etype is SkipCheckpoint:
                    return True
                elif etype is None:
                    if self.checkpoint_obj.checkpoint_fullfname is not None:
                        if self.checkpoint_obj.this_command not in self.checkpoint_obj.checkpoints_ready:
                            self.checkpoint_obj.checkpoints_ready[self.checkpoint_obj.this_command] = dict()
                        if this_checkpoint not in self.checkpoint_obj.checkpoints_ready[self.checkpoint_obj.this_command]:
                            self.checkpoint_obj.checkpoints_ready[self.checkpoint_obj.this_command][this_checkpoint] = 0
                        self.checkpoint_obj.checkpoints_ready[self.checkpoint_obj.this_command][this_checkpoint] += 1
                        with open(self.checkpoint_obj.checkpoint_fullfname, 'wb') as f:
                            pickle.dump(self.checkpoint_obj.checkpoints_ready, f, protocol=2)
        return Context(self)

#########################################
class FullVolume(object):
    
    #########################################
    def __init__(self, data_fullfname):
        self.data_fullfname = data_fullfname
        self.data = None
        if self.data_fullfname is None:
            raise NotImplementedError('Non-file preprocessed data method not implemented.')
    
    #########################################
    def create(self, config_data, volume_shape):
        if self.data_fullfname is not None:
            with h5py.File(self.data_fullfname, 'w') as data_f:
                data_f.attrs['config'] = json.dumps(config_data)
                for scale in range(config_data['num_downsamples']+1):
                    new_shape = downscales.predict_new_shape(volume_shape, scale)
                    data_f.create_dataset('volume/scale_{}'.format(scale), new_shape, dtype=np.uint16, chunks=None)
                    data_f['volume/scale_{}'.format(scale)].attrs['scale'] = scale
                data_f.create_dataset('hashes', [volume_shape[0], config_data['hash_function']['params']['hash_size']], dtype=np.float32, chunks=None)
    
    #########################################
    def load(self):
        if self.data_fullfname is not None:
            self.data = h5py.File(self.data_fullfname, 'r+')
    
    #########################################
    def get_config(self):
        if self.data_fullfname is not None:
            return load_preprocess_config_data(json.loads(self.data.attrs['config']))

    #########################################
    def get_shape(self):
        return self.data['volume/scale_0'].shape
    
    #########################################
    def get_dtype(self):
        return self.data['volume/scale_0'].dtype
    
    #########################################
    def get_hashes_dtype(self):
        return self.data['hashes'].dtype
    
    #########################################
    def get_scale_array(self, scale):
        return self.data['volume/scale_{}'.format(scale)]
    
    #########################################
    def get_scales(self):
        return { self.data['volume/{}'.format(name)].attrs['scale'] for name in self.data['volume'].keys() }
    
    #########################################
    def get_scale_arrays(self, scales=None):
        if scales is None:
            scales = self.get_scales()
        return {
                scale: self.data['volume/scale_{}'.format(scale)]
                for scale in set(scales)
            }

    #########################################
    def get_hashes_array(self):
        return self.data['hashes']
    
    #########################################
    def close(self):
        if self.data is not None:
            self.data.close()
            self.data = None

#########################################
class TrainingSet(object):
    
    #########################################
    def __init__(self, data_fullfname):
        self.data_fullfname = data_fullfname
        self.data = None
    
    #########################################
    def create(self, num_items, feature_size):
        if self.data_fullfname is not None:
            with h5py.File(self.data_fullfname, 'w') as data_f:
                data_f.create_dataset('labels', [num_items], dtype=np.uint8, chunks=None)
                data_f.create_dataset('features', [num_items, feature_size], dtype=np.float32, chunks=None)
        else:
            self.data = {
                    'labels': np.empty([num_items], dtype=np.uint8),
                    'features': np.empty([num_items, feature_size], dtype=np.float32)
                }
    
    #########################################
    def load(self):
        if self.data_fullfname is not None:
            self.data = h5py.File(self.data_fullfname, 'r+')
    
    #########################################
    def get_labels_array(self):
        return self.data['labels']
    
    #########################################
    def get_features_array(self):
        return self.data['features']
    
    #########################################
    def get_sample(self, max_sample_size_per_label):
        label_locations = dict()
        for (i, label) in enumerate(self.data['labels'][:].tolist()):
            if label < first_control_label:
                if label not in label_locations:
                    label_locations[label] = list()
                label_locations[label].append(i)
        num_labels = len(label_locations)
        label_amounts = [ len(label_locations[label]) for label in range(num_labels) ]
        
        for label in range(num_labels):
            r = random.Random(0)
            r.shuffle(label_locations[label])
        
        all_locations = [ location for label in range(num_labels) for location in label_locations[label][:max_sample_size_per_label] ]
        all_locations.sort()
        total_items_samples = len(all_locations)
        
        new_trainingset = TrainingSet(None)
        new_trainingset.create(total_items_samples, self.data['features'].shape[1])
        new_trainingset.get_labels_array()[:] = self.data['labels'][all_locations]
        new_trainingset.get_features_array()[:] = self.data['features'][all_locations,:]
        
        return new_trainingset
    
    #########################################
    def close(self):
        if self.data is not None:
            self.data.close()
            self.data = None

#########################################
class EvaluationResultsFile(object):
    
    #########################################
    def __init__(self, results_fullfname):
        self.results_fullfname = results_fullfname
    
    #########################################
    def create(self, labels):
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'w', encoding='utf-8') as f:
                print('slice', *labels, 'featurisation duration (s)', 'prediction duration (s)', sep='\t', file=f)
    
    #########################################
    def append(self, subvolume_fullfname, ious, featuriser_duration, classifier_duration):
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'a', encoding='utf-8') as f:
                print(subvolume_fullfname, *[ ('{:.3%}'.format(iou) if iou is not None else '') for iou in ious ], '{:.1f}'.format(featuriser_duration), '{:.1f}'.format(classifier_duration), sep='\t', file=f)

#########################################
class VolumeData(object):
    
    #########################################
    def __init__(self, fullfnames, shape):
        self.fullfnames = fullfnames
        self.shape = shape

#########################################
class LabelData(object):
    
    #########################################
    def __init__(self, fullfnames, shape, name):
        self.fullfnames = fullfnames
        self.shape = shape
        self.name = name

#########################################
def load_image(image_dir):
    img_data = None
    if image_dir.endswith('.jp2'):
        with tempfile.TemporaryDirectory(dir='/tmp/') as tmp_dir: #Does not work on Windows!
            subprocess.run([ 'opj_decompress', '-i', image_dir, '-o', os.path.join(tmp_dir, 'tmp.tif') ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) #Uncompressed image output for speed.
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
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        skimage.io.imsave(image_dir, image_data)

#########################################
def load_labels(labels_data):
    slice_size = np.prod(labels_data[0].shape).tolist()
    label_fullfnames = { label_data.name: label_data.fullfnames for label_data in labels_data }
    labels = sorted(label_fullfnames.keys())
    if len(labels) != len(labels_data):
        raise DataException('Some labels were declared more than once ([{}]).'.format(', '.join(label for (label, freq) in collections.Counter(label_data.name for label_data in labels_data) if freq > 1)))
    num_slices = len(label_fullfnames[labels[0]])
    subvolume_slice_labels = np.full([ slice_size*num_slices ], UNINIT_LABEL, np.uint8)
    
    for i in range(num_slices):
        slice_labels = np.full([ slice_size ], UNINIT_LABEL, np.uint8)
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
    
    labels_found = { labels[label_index] for label_index in set(np.unique(subvolume_slice_labels).tolist()) - { UNINIT_LABEL, MULTILABEL } }
    if len(labels_found) != len(labels):
        raise DataException('Labelled slices provided do not cover all labels given (missing=[{}]).'.format(', '.join(sorted(set(labels) - labels_found))))
    
    return subvolume_slice_labels

#########################################
def get_subvolume_slice_label_mask(subvolume_slice_labels):
    return np.logical_and(subvolume_slice_labels != UNINIT_LABEL, subvolume_slice_labels != MULTILABEL)

#########################################
def load_volume_dir(volume_dir):
    if not files.fexists(volume_dir):
        raise DataException('Volume directory does not exist.')
        
    volume_fullfnames = []
    with os.scandir(volume_dir) as it:
        for entry in it:
            if entry.name.startswith('.'):
                continue
            if not any(entry.name.endswith(ext) for ext in image_exts):
                continue
            if entry.is_file():
                volume_fullfnames.append(os.path.join(volume_dir, entry.name))
    if len(volume_fullfnames) == 0:
        raise DataException('Volume directory does not have any images.')
    volume_fullfnames.sort()
    
    slice_shape = None
    for (i, fullfname) in enumerate(volume_fullfnames):
        with PIL.Image.open(fullfname) as f:
            shape = (f.height, f.width)
            if f.mode[0] not in 'LI':
                raise DataException('Found volume slice that is not a greyscale image ({}).'.format(fullfname))
        if slice_shape is not None:
            if shape != slice_shape:
                raise DataException('Found differently shaped volume slices ({} and {}).'.format(volume_fullfnames[0], fullfname))
        else:
            slice_shape = shape
        
    return VolumeData(volume_fullfnames, slice_shape)

#########################################
def get_volume_slice_indexes_in_subvolume(volume_hashes, subvolume_hashes):
    #Nearest neighbour search with Manhattan distance.
    indexes = []
    for i in range(subvolume_hashes.shape[0]):
        volume_index = np.argmin(np.sum(np.abs(volume_hashes - subvolume_hashes[i, :]), axis=1)).tolist()
        indexes.append(volume_index)
    return indexes

#########################################
def load_label_dir(label_dir):
    if not files.fexists(label_dir):
        raise DataException('Label directory does not exist.')
    
    label_name = os.path.split(label_dir)[1]
    
    label_fullfnames = []
    with os.scandir(label_dir) as it:
        for entry in it:
            if entry.name.startswith('.'):
                continue
            if not any(entry.name.endswith(ext) for ext in image_exts):
                continue
            if entry.is_file():
                label_fullfnames.append(os.path.join(label_dir, entry.name))
    if len(label_fullfnames) == 0:
        raise DataException('Label directory does not have any images.')
    label_fullfnames.sort()
    
    slice_shape = None
    for (i, fullfname) in enumerate(label_fullfnames):
        with PIL.Image.open(fullfname) as f:
            shape = (f.height, f.width)
            if f.mode[0] not in 'LI':
                raise DataException('Found label slice that is not a greyscale image ({}).'.format(fullfname))
        if slice_shape is not None:
            if shape != slice_shape:
                raise DataException('Found differently shaped label slices ({} and {}).'.format(label_fullfnames[0], fullfname))
        else:
            slice_shape = shape
        
    return LabelData(label_fullfnames, slice_shape, label_name)

#########################################
def validate_annotation_data(full_volume, subvolume_data, labels_data):
    if subvolume_data.shape != full_volume.get_shape()[1:]:
        raise DataException('Subvolume slice shapes do not match volume slice shapes (volume={}, subvolume={}).'.format(full_volume.get_shape(), subvolume_data.shape))
    
    if len(labels_data) > first_control_label:
        raise DataException('Labels directory has too many labels ({}). Must be less than or equal to {}.'.format(len(labels_data), first_control_label-1))
    
    for label_data in labels_data:
        if label_data.shape != full_volume.get_shape()[1:]:
            raise DataException('Label {} slice shapes do not match volume slice shapes (volume={}, label={}).'.format(label_data.name, full_volume.get_shape()[1:], label_data.shape))
        if len(label_data.fullfnames) != len(subvolume_data.fullfnames):
            raise DataException('Number of label slices ({}) in label {} does not equal number of slices in subvolume ({}).'.format(len(label_data.fullfnames), label_data.name, len(subvolume_data.fullfnames)))

#########################################
def load_preprocess_config_file(config_fullfname):
    with open(config_fullfname, 'r', encoding='utf-8') as f:
        raw_config = json.load(f)
    return load_preprocess_config_data(raw_config)

#########################################
def load_preprocess_config_data(config_data):
    if type(config_data) is not dict:
        raise DataException('Configuration is invalid as it is not in dictionary format.')
    if set(config_data.keys()) != { 'num_downsamples', 'downsample_filter', 'hash_function' }:
        raise DataException('Configuration is invalid as it does not have the expected key values.')
    if True:
        if type(config_data['num_downsamples']) is not int:
            raise DataException('Configuration is invalid num_downsamples is not an integer.')
        if config_data['num_downsamples'] < 0:
            raise DataException('Configuration is invalid as num_downsamples is negative.')
        
        if type(config_data['downsample_filter']) is not dict:
            raise DataException('Configuration is invalid as downsample_filter is not in dictionary format.')
        if set(config_data['downsample_filter'].keys()) != { 'type', 'params' }:
            raise DataException('Configuration is invalid as downsample_filter does not have the expected key values.')
        if True:
            if type(config_data['downsample_filter']['type']) is not str:
                raise DataException('Configuration is invalid as downsample_filter type is not a string.')
            if config_data['downsample_filter']['type'] not in available_downsample_filter_types:
                raise DataException('Configuration is invalid as it declares an unexpected downsample_filter type.')
            
            if type(config_data['downsample_filter']['params']) is not dict:
                raise DataException('Configuration is invalid as downsample_filter params is not in dictionary format.')
            if config_data['downsample_filter']['type'] == 'gaussian':
                if set(config_data['downsample_filter']['params'].keys()) != { 'sigma' }:
                    raise DataException('Configuration is invalid as downsample_filter params does not have the expected key values for a downsample_filter type of {}.'.format(config_data['downsample_filter']['type']))
                if type(config_data['downsample_filter']['params']['sigma']) is not float:
                    raise DataException('Configuration is invalid as downsample_filter params sigma is not a floating point number.')
            elif config_data['downsample_filter']['type'] == 'null':
                pass
            else:
                NotImplementedError('Downsample filter {} is not implemented.'.format(config_data['downsample_filter']['type']))

        if type(config_data['hash_function']) is not dict:
            raise DataException('Configuration is invalid as hash_function is not in dictionary format.')
        if set(config_data['hash_function'].keys()) != { 'type', 'params' }:
            raise DataException('Configuration is invalid as hash_function does not have the expected key values.')
        if True:
            if type(config_data['hash_function']['type']) is not str:
                raise DataException('Configuration is invalid as hash_function type is not a string.')
            if config_data['hash_function']['type'] not in available_hash_functions:
                raise DataException('Configuration is invalid as it declares an unexpected hash_function type.')
            
            if type(config_data['hash_function']['params']) is not dict:
                raise DataException('Configuration is invalid as hash_function params is not in dictionary format.')
            if config_data['hash_function']['type'] == 'random_indexing':
                if set(config_data['hash_function']['params'].keys()) != { 'hash_size' }:
                    raise DataException('Configuration is invalid as hash_function params does not have the expected key values for a hash_function type of {}.'.format(config_data['hash_function']['type']))
                
                if type(config_data['hash_function']['params']['hash_size']) is not int:
                    raise DataException('Configuration is invalid as hash_function params hash_size is not an integer.')
                if config_data['hash_function']['params']['hash_size'] <= 0:
                    raise DataException('Configuration is invalid as hash_function params hash_size is not a positive integer.')
            else:
                NotImplementedError('Hash function {} is not implemented.'.format(config_data['hash_function']['type']))
    
    downsample_filter = {
            'gaussian': downscales.GaussianDownsampleKernel(**config_data['downsample_filter']['params']),
            'null': downscales.NullDownsampleKernel(),
        }[config_data['downsample_filter']['type']]
    
    hash_function = {
            'random_indexing': hashfunctions.RandomIndexingHashFunction(**config_data['hash_function']['params']),
        }[config_data['hash_function']['type']]
    
    return (config_data, config_data['num_downsamples'], downsample_filter, hash_function)

#########################################
def load_train_config_file(config_fullfname, full_volume=None):
    with open(config_fullfname, 'r', encoding='utf-8') as f:
        raw_config = json.load(f)
    return load_train_config_data(raw_config, full_volume)

#########################################
def load_train_config_data(config_data, full_volume=None):
    if type(config_data) is not dict:
        raise DataException('Configuration is invalid as it is not in dictionary format.')
    if set(config_data.keys()) != { 'featuriser', 'classifier', 'training_set' }:
        raise DataException('Configuration is invalid as it does not have the expected key values.')
    
    if type(config_data['featuriser']) is not dict:
        raise DataException('Configuration is invalid as featuriser is not in dictionary format.')
    if set(config_data['featuriser'].keys()) != { 'type', 'params' }:
        raise DataException('Configuration is invalid as featuriser does not have the expected key values.')
    if True:
        if type(config_data['featuriser']['type']) is not str:
            raise DataException('Configuration is invalid as featuriser type is not a string.')
        if config_data['featuriser']['type'] not in available_featurisers:
            raise DataException('Configuration is invalid as it declares an unexpected featuriser type.')
        
        featuriser_params = None
        if type(config_data['featuriser']['params']) is not dict:
            raise DataException('Configuration is invalid as featuriser params is not in dictionary format.')
        if config_data['featuriser']['type'] == 'histograms':
            if set(config_data['featuriser']['params'].keys()) != { 'use_voxel_value', 'histograms' }:
                raise DataException('Configuration is invalid as featuriser params does not have the expected key values for a featuriser type of {}.'.format(config_data['featuriser']['type']))
            
            if type(config_data['featuriser']['params']['use_voxel_value']) is not str:
                raise DataException('Configuration is invalid as featuriser params use_voxel_value is not a string.')
            if config_data['featuriser']['params']['use_voxel_value'] not in { 'yes', 'no' }:
                raise DataException('Configuration is invalid as featuriser params use_voxel_value is not \'yes\' or \'no\'.')
            
            if type(config_data['featuriser']['params']['histograms']) is not list:
                raise DataException('Configuration is invalid as featuriser params histograms is not a list.')
            if len(config_data['featuriser']['params']['histograms']) == 0:
                raise DataException('Configuration is invalid as featuriser params histograms is empty.')
            for (i, entry) in enumerate(config_data['featuriser']['params']['histograms']):
                if type(entry) is not dict:
                    raise DataException('Configuration is invalid as featuriser params histograms entry {} is not in dictionary format.'.format(i))
                if set(entry.keys()) != { 'radius', 'scale', 'num_bins' }:
                    raise DataException('Configuration is invalid as featuriser params histograms entry {} does not have the expected key values.'.format(i))
                
                if type(entry['radius']) is not int:
                    raise DataException('Configuration is invalid as featuriser params entry {} radius is not an integer.'.format(i))
                
                if type(entry['scale']) is not int:
                    raise DataException('Configuration is invalid as featuriser params entry {} scale is not an integer.'.format(i))
                
                if type(entry['num_bins']) is not int:
                    raise DataException('Configuration is invalid as featuriser params entry {} num_bins is not an integer.'.format(i))
            
            featuriser_params = {
                    'use_voxel_value': config_data['featuriser']['params']['use_voxel_value'] == 'yes',
                    'histogram_params': [
                            (entry['radius'], entry['scale'], entry['num_bins'])
                            for entry in config_data['featuriser']['params']['histograms']
                        ]
                }
        else:
            NotImplementedError('Featuriser {} is not implemented.'.format(config_data['featuriser']))
    
    if type(config_data['classifier']) is not dict:
        raise DataException('Configuration is invalid as classifier is not in dictionary format.')
    if set(config_data['classifier'].keys()) != { 'type', 'params' }:
        raise DataException('Configuration is invalid as classifier does not have the expected key values.')
    if True:
        if type(config_data['classifier']['type']) is not str:
            raise DataException('Configuration is invalid as classifier type is not a string.')
        if config_data['classifier']['type'] not in available_classifiers:
            raise DataException('Configuration is invalid as it declares an unexpected classifier type.')
        
        if type(config_data['classifier']['params']) is not dict:
            raise DataException('Configuration is invalid as classifier params is not in dictionary format.')
        if config_data['classifier']['type'] == 'random_forest':
            if set(config_data['classifier']['params'].keys()) != { 'n_estimators', 'max_depth', 'min_samples_leaf' }:
                raise DataException('Configuration is invalid as classifier params does not have the expected key values for a classifier type of {}.'.format(config_data['classifier']['type']))
                
            if type(config_data['classifier']['params']['n_estimators']) is not int:
                raise DataException('Configuration is invalid as classifier params n_estimators is not an integer.')
            
            if type(config_data['classifier']['params']['max_depth']) is not int:
                raise DataException('Configuration is invalid as classifier params max_depth is not an integer.')
            
            if type(config_data['classifier']['params']['min_samples_leaf']) is not int:
                raise DataException('Configuration is invalid as classifier params min_samples_leaf is not an integer.')
        else:
            NotImplementedError('Classifier {} is not implemented.'.format(config_data['classifier']['type']))
    
    if type(config_data['training_set']) is not dict:
        raise DataException('Configuration is invalid as training_set is not in dictionary format.')
    if set(config_data['training_set'].keys()) != { 'sample_size_per_label' }:
        raise DataException('Configuration is invalid as training_set does not have the expected key values.')
    if True:
        if type(config_data['training_set']['sample_size_per_label']) is not int:
            raise DataException('Configuration is invalid as training_set sample_size_per_label is not an integer.')
        if config_data['training_set']['sample_size_per_label'] == 0 or config_data['training_set']['sample_size_per_label'] < -1:
            raise DataException('Configuration is invalid as training_set sample_size_per_label is 0 or a negative number other than -1.')

    featuriser = {
            'histograms': featurisers.HistogramFeaturiser(**featuriser_params),
        }[config_data['featuriser']['type']]
    
    classifier = {
            'random_forest': sklearn.ensemble.RandomForestClassifier(**config_data['classifier']['params'], random_state=0),
        }[config_data['classifier']['type']]
    
    if full_volume is not None:
        if not (featuriser.get_scales_needed() <= full_volume.get_scales()):
            raise DataException('Featuriser requires scales that are not included in preprocessed volume (missing scales=[{}]).'.format(', '.join(sorted(featuriser.get_scales_needed() - full_volume.get_scales()))))
    
    return (config_data, featuriser, classifier)

#########################################
def load_model_file(model_fullfname, full_volume=None):
    if not files.fexists(model_fullfname):
        raise DataException('Model does not exist.')
    
    with open(model_fullfname, 'rb') as f:
        model_data = pickle.load(f)
    
    return load_model_data(model_data)

#########################################
def load_model_data(model_data, full_volume=None):
    if type(model_data) is not dict:
        raise DataException('Model is invalid as it is not in dictionary format.')
    if set(model_data.keys()) != { 'model', 'labels', 'config' }:
        raise DataException('Model is invalid as it does not have the expected key values.')
    
    (config_data, featuriser, _) = load_train_config_data(model_data['config'], full_volume)
    
    if type(model_data['labels']) is not list:
        raise DataException('Model is invalid as labels are not a list.')
    for (i, entry) in enumerate(model_data['labels']):
        if type(entry) is not str:
            raise DataException('Model is invalid as label entry {} is not a string.'.format(i))
    
    if config_data['classifier'] == 'random_forest':
        if not isinstance(model_data['model'], sklearn.ensemble.RandomForestClassifier):
            raise DataException('Model is invalid as it is not a random forest as declared.')
    if model_data['model'].n_classes_ != len(model_data['labels']):
        raise DataException('Model is invalid as the number of classes is not as declared (declared={}, actual={}).'.format(len(model_data['labels']), model_data['model'].n_classes_))
    if config_data['featuriser'] == 'histograms':
        if model_data['model'].n_features_ != featuriser.get_feature_size():
            raise DataException('Model is invalid as the number of features is not as declared (declared={}, actual={}).'.format(featuriser.get_feature_size(), model_data['model'].n_features_))
    
    if full_volume is not None:
        if not (featuriser.get_scales_needed() <= full_volume.get_scales()):
            raise DataException('Featuriser requires scales that are not included in preprocessed volume (missing scales=[{}]).'.format(', '.join(sorted(featuriser.get_scales_needed() - full_volume.get_scales()))))
    
    return (model_data['labels'], config_data, featuriser, model_data['model'])

#########################################
def save_model(model_fullfname, model):
    if model_fullfname is not None:
        with open(model_fullfname, 'wb') as f:
            pickle.dump(model, f, protocol=2)

#########################################
def check_preprocessed_filename(data_fullfname, must_exist=False):
    dir = os.path.split(data_fullfname)[0]
    if dir != '' and not files.fexists(dir):
        raise DataException('Preprocessed file\'s directory does not exist.')
    if not data_fullfname.endswith('.hdf'):
        raise DataException('Preprocessed file\'s file name does not end with .hdf.')
    if must_exist and not files.fexists(data_fullfname):
        raise DataException('Preprocessed file does not exist.')

#########################################
def check_checkpoint_filename(checkpoint_fullfname, must_exist=False):
    dir = os.path.split(checkpoint_fullfname)[0]
    if dir != '' and not files.fexists(dir):
        raise DataException('Checkpoint directory does not exist.')
    if not checkpoint_fullfname.endswith('.pkl'):
        raise DataException('Checkpoint file name does not end with .pkl.')
    if must_exist and not files.fexists(checkpoint_fullfname):
        raise DataException('Checkpoint file does not exist.')

#########################################
def check_model_filename(model_fullfname, must_exist=False):
    dir = os.path.split(model_fullfname)[0]
    if dir != '' and not files.fexists(dir):
        raise DataException('Model file\'s directory does not exist.')
    if not model_fullfname.endswith('.pkl'):
        raise DataException('Model file\'s file name does not end with .pkl.')
    if must_exist and not files.fexists(model_fullfname):
        raise DataException('Model file does not exist.')

#########################################
def check_evaluation_results_filename(evaluation_results_fullfname, must_exist=False):
    dir = os.path.split(evaluation_results_fullfname)[0]
    if dir != '' and not files.fexists(dir):
        raise DataException('Result file\'s directory does not exist.')
    if not evaluation_results_fullfname.endswith('.txt'):
        raise DataException('Result file\'s file name does not end with .txt.')
    if must_exist and not files.fexists(evaluation_results_fullfname):
        raise DataException('Result file does not exist.')

#########################################
def check_segmentation_results_directory(results_dir):
    if not files.fexists(results_dir):
        raise DataException('Results directory does not exist.')
