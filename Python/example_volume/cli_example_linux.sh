#!/bin/sh
#
# Copyright © 2020 Marc Tanti
#
# This file is part of ASEMI-segmenter.

../bin/preprocess.py \
    --volume_dir "volume" \
    --config_fullfname "configs/preprocess_config.json" \
    --result_data_fullfname "output/preprocess/volume.hdf" \
    --max_processes 4 \
    --max_batch_memory 1.0

../bin/analyse_data.py \
    --subvolume_dir "training_set/subvolume" \
    --label_dirs \
        "training_set/labels/air" \
        "training_set/labels/tissues" \
        "training_set/labels/bones" \
    --config "configs/analyse_data_config.json" \
    --highlight_radius 3 \
    --results_dir "output/analyse_data"

../bin/tune.py \
    --preproc_volume_fullfname "output/preprocess/volume.hdf" \
    --train_subvolume_dir "training_set/subvolume" \
    --train_label_dirs \
        "training_set/labels/air" \
        "training_set/labels/tissues" \
        "training_set/labels/bones" \
    --eval_subvolume_dir "validation_set/subvolume" \
    --eval_label_dirs \
        "validation_set/labels/air" \
        "validation_set/labels/tissues" \
        "validation_set/labels/bones" \
    --config_fullfname "configs/tune_config.json" \
    --search_results_fullfname "output/tune/results.txt" \
    --best_result_fullfname "output/tune/result.json" \
    --features_table_fullfname "output/tune/features.hdf" \
    --max_processes 4 \
    --max_batch_memory 1.0 \

../bin/train.py \
    --preproc_volume_fullfname "output/preprocess/volume.hdf" \
    --subvolume_dir "training_set/subvolume" \
    --label_dirs \
        "training_set/labels/air" \
        "training_set/labels/tissues" \
        "training_set/labels/bones" \
    --config_fullfname "configs/train_config.json" \
    --result_segmenter_fullfname "output/train/segmenter.pkl" \
    --trainingset_file_fullfname "output/train/trainingset.hdf" \
    --max_processes 4 \
    --max_batch_memory 1.0

../bin/evaluate.py \
    --segmenter_fullfname "output/train/segmenter.pkl" \
    --preproc_volume_fullfname "output/preprocess/volume.hdf" \
    --subvolume_dir "testing_set/subvolume" \
    --label_dirs \
        "testing_set/labels/air" \
        "testing_set/labels/tissues" \
        "testing_set/labels/bones" \
    --results_dir "output/evaluate" \
    --max_processes 4 \
    --max_batch_memory 1.0

../bin/segment.py \
    --segmenter_fullfname "output/train/segmenter.pkl" \
    --preproc_volume_fullfname "output/preprocess/volume.hdf" \
    --config_fullfname "configs/segment_config.json" \
    --results_dir "output/segment" \
    --max_processes 4 \
    --max_batch_memory 1.0
