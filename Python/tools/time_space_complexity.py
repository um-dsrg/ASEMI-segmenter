import numpy as np
import memory_profiler
import tempfile
import shutil
import os
import sys
from asemi_segmenter.lib import regions
from asemi_segmenter.lib import times
from asemi_segmenter.lib import files
from asemi_segmenter.lib import images
from asemi_segmenter.lib import featurisers
from asemi_segmenter import preprocess
from asemi_segmenter import train
from asemi_segmenter import segment
from asemi_segmenter import listener

class Listener(listener.ProgressListener):
    def error_output(self, text):
        raise Exception(text)

#########################################
def measure(side_length, num_training_slices, num_labels, num_runs, train_config, max_processes, max_batch_memory, listener=lambda run, train_time, train_memory, segment_time, segment_memory:None):
    featuriser = featurisers.load_featuriser_from_config(train_config['featuriser'])
    preprocess_config = {
            'num_downsamples': max(featuriser.get_scales_needed()),
            'downsample_filter': {
                    'type': 'none',
                    'params': {
                        }
                },
            'hash_function': {
                    'type': 'random_indexing',
                    'params': {
                            'hash_size': 10
                        }
                }
        }
    del featuriser
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print('Creating full volume slices')
        files.mkdir(os.path.join(temp_dir, 'volume'))
        r = np.random.RandomState(0)
        for i in range(side_length):
            images.save_image(os.path.join(temp_dir, 'volume', '{}.tif'.format(i)), r.randint(0, 2**16, size=[side_length,side_length], dtype=np.uint16))
        
        print('Creating sub volume slices')
        files.mkdir(os.path.join(temp_dir, 'subvolume'))
        for i in range(0, side_length-side_length%num_training_slices, side_length//num_training_slices):
            shutil.copy(os.path.join(temp_dir, 'volume', '{}.tif'.format(i)), os.path.join(temp_dir, 'subvolume', '{}.tif'.format(i)))
        
        print('Creating label slices')
        files.mkdir(os.path.join(temp_dir, 'labels'))
        r = np.random.RandomState(0)
        for label in range(num_labels):
            files.mkdir(os.path.join(temp_dir, 'labels', '_{}'.format(label)))
            for i in range(0, side_length-side_length%num_training_slices, side_length//num_training_slices):
                images.save_image(os.path.join(temp_dir, 'labels', '_{}'.format(label), '{}.tif'.format(i)), r.randint(0, 2, size=[side_length,side_length], dtype=np.uint16))
        
        print('Preprocessing volume')
        preprocess.main(
                volume_dir=os.path.join(temp_dir, 'volume'),
                config=preprocess_config,
                result_data_fullfname=os.path.join(temp_dir, 'full_volume.hdf'),
                checkpoint_fullfname=None,
                restart_checkpoint=False,
                max_processes=max_processes,
                max_batch_memory=max_batch_memory,
                listener=Listener()
            )
        
        files.mkdir(os.path.join(temp_dir, 'segmentation'))
        
        print('Starting measurements')
        for run in range(1, num_runs+1):
            print('Measuring run', run)
            with times.Timer() as timer:
                mem_usage = memory_profiler.memory_usage(
                    (
                        train.main,
                        [],
                        dict(
                            preproc_volume_fullfname=os.path.join(temp_dir, 'full_volume.hdf'),
                            subvolume_dir=os.path.join(temp_dir, 'subvolume'),
                            label_dirs=[ os.path.join(temp_dir, 'labels', '_{}'.format(label)) for label in range(num_labels) ],
                            config=train_config,
                            result_segmenter_fullfname=os.path.join(temp_dir, 'segmenter.pkl'),
                            trainingset_file_fullfname=None,
                            checkpoint_fullfname=None,
                            restart_checkpoint=False,
                            max_processes=max_processes,
                            max_batch_memory=max_batch_memory,
                            listener=Listener()
                        )
                    )
                )
                
            train_time = timer.duration
            train_memory = max(mem_usage)
            
            with times.Timer() as timer:
                mem_usage = memory_profiler.memory_usage(
                    (
                        segment.main,
                        [],
                        dict(
                            segmenter=os.path.join(temp_dir, 'segmenter.pkl'),
                            preproc_volume_fullfname=os.path.join(temp_dir, 'full_volume.hdf'),
                            config={'soft_segmentation': 'no'},
                            results_dir=os.path.join(temp_dir, 'segmentation'),
                            checkpoint_fullfname=None,
                            restart_checkpoint=False,
                            max_processes=max_processes,
                            max_batch_memory=max_batch_memory,
                            listener=Listener()
                        )
                    )
                )
            segment_time = timer.duration
            segment_memory = max(mem_usage)
            
            listener(run, train_time, train_memory, segment_time, segment_memory)
        
        print()


num_training_slices = 10
num_labels = 5
num_runs = 3
train_config = {
    'featuriser': {
        'type': 'composite',
        'params': {
            'featuriser_list': [
                {
                    'type': 'voxel',
                    'params': {}
                    },
                {
                    'type': 'histogram',
                    'params': {
                        'radius': 1,
                        'scale': 0,
                        'num_bins': 32
                        }
                    },
                {
                    'type': 'histogram',
                    'params': {
                        'radius': 20,
                        'scale': 0,
                        'num_bins': 32
                        }
                        }
                ]
            }
        },
    'classifier': {
            'type': 'random_forest',
            'params': {
                'n_estimators': 32,
                'max_depth': 10,
                'min_samples_leaf': 1
                }
        },
    'training_set': {
        'sample_size_per_label': -1
        }
    }

with open('results.txt', 'w', encoding='utf-8') as f:
    print('max_batch_memory', 'max_processes', 'side_length', 'slice_area', 'cube_volume', 'run', 'train_time', 'train_memory', 'segment_time', 'segment_memory', sep='\t', file=f)

for max_batch_memory in [ 0.1, 0.5, 1.0 ]:
    for max_processes in [ 1, 2, 4 ]:
        for side_length in [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 ]:
            print('max_batch_memory', max_batch_memory, '-', 'max_processes', max_processes, '-', 'side_length', side_length)
            def listener(run, train_time, train_memory, segment_time, segment_memory):
                with open('results.txt', 'a', encoding='utf-8') as f:
                    print(max_batch_memory, max_processes, side_length, side_length**2, side_length**3, run, train_time, train_memory, segment_time, segment_memory, sep='\t', file=f)
                
            measure(side_length, num_training_slices, num_labels, num_runs, train_config, max_processes, max_batch_memory, listener)

input('Press enter to exit.')
