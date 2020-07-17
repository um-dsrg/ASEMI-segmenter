#!/bin/bash

python time_space_complexity.py \
    --train_config_fullfname "train_config.json" \
    --segment_config_fullfname "segment_config.json" \
    --results_fullfname "results.txt" \
    --max_processes_featuriser 1 2 4 \
    --max_processes_classifier 1 2 4 \
    --max_batch_memory 0.1 1.0 10.0 \
    --side_length 100 200 300 400 500 \
    --num_training_slices 10 \
    --num_labels 5 \
    --num_runs 3 \
    --num_simultaneous_slices 2 \
    --voxel_seed 0 \
    --label_seed 0 \
    --use_gpu "no"
