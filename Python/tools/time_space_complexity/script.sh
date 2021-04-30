#!/bin/bash
#
# Copyright © 2020 Marc Tanti
#
# This file is part of ASEMI-segmenter.
#
# ASEMI-segmenter is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ASEMI-segmenter is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASEMI-segmenter.  If not, see <http://www.gnu.org/licenses/>.

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
