'''Analyse command.'''

import os
import json
import numpy as np
from asemi_segmenter.listener import ProgressListener
from asemi_segmenter.lib import datasets
from asemi_segmenter.lib import images
from asemi_segmenter.lib import times
from asemi_segmenter.lib import validations
from asemi_segmenter.lib import volumes
from asemi_segmenter.lib import colours


#########################################
def _loading_data(
        subvolume_dir, label_dirs, config, highlight_radius,
        results_dir, listener
    ):
    '''Loading data stage.'''
    listener.log_output('> Subvolume')
    listener.log_output('>> {}'.format(subvolume_dir))
    subvolume_data = volumes.load_volume_dir(subvolume_dir)
    subvolume_fullfnames = subvolume_data.fullfnames
    
    listener.log_output('> Labels')
    labels_data = []
    for label_dir in label_dirs:
        listener.log_output('>> {}'.format(label_dir))
        label_data = volumes.load_label_dir(label_dir)
        labels_data.append(label_data)
        listener.log_output('>>> {}'.format(label_data.name))
    labels = sorted(label_data.name for label_data in labels_data)
    validations.validate_annotation_data(None, subvolume_data, labels_data)
    
    listener.log_output('> Config')
    if isinstance(config, str):
        listener.log_output('>> {}'.format(config))
        validations.check_filename(config, '.json', True)
        with open(config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    else:
        config_data = config
    validations.validate_json_with_schema_file(config_data, 'dataset.json')
    if 'samples_to_skip_per_label' not in config_data['dataset']:
        config_data['dataset']['samples_to_skip_per_label'] = 0
    
    listener.log_output('> Highlight radius')
    listener.log_output('>> {}'.format(highlight_radius))
    if highlight_radius < 1:
        raise ValueError('Highlight radius must be positive.')
    
    listener.log_output('> Result')
    listener.log_output('>> {}'.format(results_dir))
    validations.check_directory(results_dir)
    
    listener.log_output('> Initialising')
    
    listener.log_output('> Other parameters:')
    listener.log_output('>> -')
    
    return (subvolume_fullfnames, labels_data, config_data)


#########################################
def _analysing(
        subvolume_fullfnames, labels_data, config_data, highlight_radius,
        results_dir, listener
    ):
    '''Analysing stage.'''
    sample_size_per_label = config_data['dataset']['sample_size_per_label']
    labels = sorted(label_data.name for label_data in labels_data)
    
    overlap_matrices = volumes.get_label_overlap(labels_data)
    for (i, overlap_matrix) in enumerate(overlap_matrices):
        with open(os.path.join(results_dir, 'overlap_slice_{}.txt'.format(i + 1)), 'w', encoding='utf-8') as f:
            print('', *labels, sep='\t', file=f)
            for label1 in labels:
                print(label1, *[overlap_matrix[label1][label2] for label2 in labels], sep='\t', file=f)
    
    if sample_size_per_label != -1:
        subvolume_slice_labels = volumes.load_labels(labels_data)
        
        (voxel_indexes, label_positions) = datasets.sample_voxels(
            subvolume_slice_labels,
            sample_size_per_label,
            len(labels),
            list(range(len(subvolume_fullfnames))),
            labels_data[0].shape,
            config_data['dataset']['samples_to_skip_per_label'],
            seed=0
            )
        
        label_palette = colours.LabelPalette(labels)
        images.save_image(
            os.path.join(results_dir, 'legend.tiff'),
            images.matplotlib_to_imagedata(label_palette.get_legend()),
            num_bits=8,
            compress=True
            )
        
        sample_labels = np.empty((len(voxel_indexes),), np.uint8)
        for (label_index, pos) in enumerate(label_positions):
            sample_labels[pos] = label_index
        subvolume_slice_colours = label_palette.label_indexes_to_colours(sample_labels).tolist()
        slice_voxel_samples = [list() for _ in subvolume_fullfnames]
        for ((slc, row, col), colour) in zip(voxel_indexes, subvolume_slice_colours):
            slice_voxel_samples[slc].append((row, col, colour))
        
        for (i, (fname, samples)) in enumerate(zip(subvolume_fullfnames, slice_voxel_samples)):
            im = images.load_image(fname, num_bits=8)
            im = im.reshape(im.shape+(1,))
            im = im.repeat(3, axis=2)
            
            r = highlight_radius
            for (row, col, colour) in samples:
                im[row-r:row+r+1, col-r:col+r+1, :] = colour
            
            images.save_image(
                os.path.join(results_dir, 'samples_slice_{}.tiff'.format(i + 1)),
                im,
                num_bits=8,
                compress=True
                )
    
    return ()


#########################################
def main(
        subvolume_dir, label_dirs, config, highlight_radius, results_dir,
        listener=ProgressListener(), debug_mode=False
    ):
    '''
    Analyse a data set (training set or evaluation set).

    :param str subvolume_dir: The path to the directory containing copies from the full
        volume slices that were labelled.
    :param list label_dirs: A list of paths to the directories containing labelled
        slices with the number of labels being equal to the number of directories and
        the number of images in each directory being equal to the number of subvolume
        images.
    :param config: The configuration to use when extracting the dataset (can be either a path to a
        json file containing the configuration or a dictionary specifying the configuration
        directly). See user guide for description of the analyse_data configuration.
    :type config: str or dict
    :param int highlight_radius: The radius of squares that mark the sampled voxels.
    :param str results_dir: The path to the directory in which to store the output files.
    :param ProgressListener listener: The command's progress listener.
    :param bool debug_mode: Whether to show full error messages or just simple ones.
    '''
    try:
        with times.Timer() as full_timer:
            listener.overall_progress_start(2)

            listener.log_output('Starting data analysis process')
            listener.log_output('')

            ###################

            listener.overall_progress_update(1, 'Loading data')
            listener.log_output(times.get_timestamp())
            listener.log_output('Loading data')
            with times.Timer() as timer:
                (subvolume_fullfnames, labels_data, config_data) = _loading_data(
                    subvolume_dir, label_dirs, config, highlight_radius,
                    results_dir, listener
                    )
            listener.log_output('Data loaded')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(2, 'Analysing')
            listener.log_output(times.get_timestamp())
            listener.log_output('Analysing')
            with times.Timer() as timer:
                () = _analysing(
                    subvolume_fullfnames, labels_data, config_data, highlight_radius,
                    results_dir, listener
                    )
            listener.log_output('Analysed')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

        listener.log_output('Done')
        listener.log_output('Entire process duration: {}'.format(
            times.get_readable_duration(full_timer.duration)
            ))
        listener.log_output(times.get_timestamp())

        listener.overall_progress_end()
    except Exception as ex:
        listener.error_output(str(ex))
        if debug_mode:
            raise
    finally:
        pass