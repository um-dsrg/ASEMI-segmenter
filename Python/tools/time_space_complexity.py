import numpy as np
import memory_profiler
import tempfile
import shutil
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from asemi_segmenter.lib import regions
from asemi_segmenter.lib import datas
from asemi_segmenter.lib import times
from asemi_segmenter import segmenter

class Listener(segmenter.ProgressListener):
    def error_output(self, text):
        raise Exception(text)

#########################################
def measure(volume_cube_side_lengths, num_training_slices, num_labels, num_runs, train_config, max_processes, max_batch_memory, listener=lambda side_length, run, train_time, train_memory, segment_time, segment_memory:None):
    (_, featuriser, _) = datas.load_train_config_data(train_config)
    preprocess_config = {
            'num_downsamples': max(featuriser.get_scales_needed()),
            'downsample_filter': {
                    'type': 'null',
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
    
    for side_length in volume_cube_side_lengths:
        with tempfile.TemporaryDirectory() as temp_dir:
            print('Creating test data of side length', side_length)
            
            print('Creating full volume slices')
            os.mkdir(os.path.join(temp_dir, 'volume'))
            r = np.random.RandomState(0)
            for i in range(side_length):
                datas.save_image(os.path.join(temp_dir, 'volume', '{}.tif'.format(i)), r.randint(0, 2**16, size=[side_length,side_length], dtype=np.uint16))
            
            print('Creating sub volume slices')
            os.mkdir(os.path.join(temp_dir, 'subvolume'))
            for i in range(0, side_length-side_length%num_training_slices, side_length//num_training_slices):
                shutil.copy(os.path.join(temp_dir, 'volume', '{}.tif'.format(i)), os.path.join(temp_dir, 'subvolume', '{}.tif'.format(i)))
            
            print('Creating label slices')
            os.mkdir(os.path.join(temp_dir, 'labels'))
            r = np.random.RandomState(0)
            for label in range(num_labels):
                os.mkdir(os.path.join(temp_dir, 'labels', '_{}'.format(label)))
                for i in range(0, side_length-side_length%num_training_slices, side_length//num_training_slices):
                    datas.save_image(os.path.join(temp_dir, 'labels', '_{}'.format(label), '{}.tif'.format(i)), r.randint(0, 2, size=[side_length,side_length], dtype=np.uint16))
            
            print('Preprocessing volume')
            segmenter.preprocess(
                    volume_dir=os.path.join(temp_dir, 'volume'),
                    config=preprocess_config,
                    result_data_fullfname=os.path.join(temp_dir, 'full_volume.hdf'),
                    checkpoint_fullfname=None,
                    restart_checkpoint=False,
                    max_processes=max_processes,
                    max_batch_memory=max_batch_memory,
                    listener=Listener()
                )
            
            os.mkdir(os.path.join(temp_dir, 'segmentation'))
            
            print('Starting measurements')
            for run in range(1, num_runs+1):
                print('Measuring run', run)
                with times.Timer() as timer:
                    mem_usage = memory_profiler.memory_usage(
                        (
                            segmenter.train,
                            [],
                            dict(
                                preproc_volume_fullfname=os.path.join(temp_dir, 'full_volume.hdf'),
                                subvolume_dir=os.path.join(temp_dir, 'subvolume'),
                                label_dirs=[ os.path.join(temp_dir, 'labels', '_{}'.format(label)) for label in range(num_labels) ],
                                config=train_config,
                                result_model_fullfname=os.path.join(temp_dir, 'model.pkl'),
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
                            segmenter.segment,
                            [],
                            dict(
                                model=os.path.join(temp_dir, 'model.pkl'),
                                preproc_volume_fullfname=os.path.join(temp_dir, 'full_volume.hdf'),
                                results_dir=os.path.join(temp_dir, 'segmentation'),
                                checkpoint_fullfname=None,
                                restart_checkpoint=False,
                                soft_segmentation=True,
                                max_processes=max_processes,
                                max_batch_memory=max_batch_memory,
                                listener=Listener()
                            )
                        )
                    )
                segment_time = timer.duration
                segment_memory = max(mem_usage)
                
                listener(side_length, run, train_time, train_memory, segment_time, segment_memory)
            
            print()

#########################################
def plot(volume_cube_side_lengths, num_training_slices, num_labels, num_runs, train_config, max_processes, max_batch_memory):
    (fig, axs) = plt.subplots(2, 2)

    [ train_time_plot ] = axs[0,0].plot([], [], color='red', linestyle='', marker='o', markersize=2)
    axs[0,0].set_title('Train time')
    axs[0,0].set_xlim(0, 1)
    axs[0,0].set_xlabel('slice area')
    axs[0,0].set_ylim(0, 1)
    axs[0,0].set_ylabel('seconds')
    axs[0,0].grid(True)
    
    [ train_mem_plot ] = axs[0,1].plot([], [], color='red', linestyle='', marker='o', markersize=2)
    axs[0,1].set_title('Train memory')
    axs[0,1].set_xlim(0, 1)
    axs[0,1].set_xlabel('slice area')
    axs[0,1].set_ylim(0, 1)
    axs[0,1].set_ylabel('MB')
    axs[0,1].grid(True)
    
    [ seg_time_plot ] = axs[1,0].plot([], [], color='red', linestyle='', marker='o', markersize=2)
    axs[1,0].set_title('Segment time')
    axs[1,0].set_xlim(0, 1)
    axs[1,0].set_xlabel('cube volume')
    axs[1,0].set_ylim(0, 1)
    axs[1,0].set_ylabel('seconds')
    axs[1,0].grid(True)
    
    [ seg_mem_plot ] = axs[1,1].plot([], [], color='red', linestyle='', marker='o', markersize=2)
    axs[1,1].set_title('Segment memory')
    axs[1,1].set_xlim(0, 1)
    axs[1,1].set_xlabel('cube volume')
    axs[1,1].set_ylim(0, 1)
    axs[1,1].set_ylabel('MB')
    axs[1,1].grid(True)

    fig.tight_layout()
    
    with open('results.txt', 'w', encoding='utf-8') as f:
        print('side_length', 'slice_area', 'cube_volume', 'run', 'train_time', 'train_memory', 'segment_time', 'segment_memory', sep='\t', file=f)
    all_slice_areas = []
    all_cube_volumes = []
    all_train_times = []
    all_train_mems = []
    all_seg_times = []
    all_seg_mems = []
    
    def listener(side_length, run, train_time, train_memory, segment_time, segment_memory):
        area = side_length**2
        volume = side_length**3
        
        with open('results.txt', 'a', encoding='utf-8') as f:
            print(side_length, area, volume, run, train_time, train_memory, segment_time, segment_memory, sep='\t', file=f)
        
        all_slice_areas.append(area)
        all_cube_volumes.append(volume)
        all_train_times.append(train_time)
        all_train_mems.append(train_memory)
        all_seg_times.append(segment_time)
        all_seg_mems.append(segment_memory)
    
        train_time_plot.set_data(all_slice_areas, all_train_times)
        train_mem_plot.set_data(all_slice_areas, all_train_mems)
        seg_time_plot.set_data(all_cube_volumes, all_seg_times)
        seg_mem_plot.set_data(all_cube_volumes, all_seg_mems)
        
        axs[0,0].set_ylim(0, max(all_train_times)*1.1)
        axs[0,0].set_xlim(0, area*1.1)
        axs[0,1].set_ylim(0, max(all_train_mems)*1.1)
        axs[0,1].set_xlim(0, area*1.1)
        axs[1,0].set_ylim(0, max(all_seg_times)*1.1)
        axs[1,0].set_xlim(0, volume*1.1)
        axs[1,1].set_ylim(0, max(all_seg_mems)*1.1)
        axs[1,1].set_xlim(0, volume*1.1)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.savefig('results.png')
    
    measure(volume_cube_side_lengths, num_training_slices, num_labels, num_runs, train_config, max_processes, max_batch_memory, listener)
    
plot(
        volume_cube_side_lengths=list(range(10, 200+1, 10)),
        num_training_slices=10,
        num_labels=5,
        num_runs=3,
        train_config={
                'featuriser': {
                        'type': 'histograms',
                        'params': {
                                'use_voxel_value': 'yes',
                                'histograms': [
                                        { 'radius': 1, 'scale': 0, 'num_bins': 32 },
                                        { 'radius': 20, 'scale': 0, 'num_bins': 32 }
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
            },
        max_processes=1,
        max_batch_memory=0.1
    )

input('Press enter to exit.')
