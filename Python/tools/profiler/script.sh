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

python profiler.py \
    --volume_fullfname "../../example_volume/output/preprocess/volume.hdf" \
    --segmenter_fullfname "../../example_volume/output/train/segmenter.pkl" \
    --segment_config_fullfname "segment_config.json" \
    --results_dir "../../example_volume/output/segment" \
    --profiler_results_fullfname "results.txt" \
    --max_processes_featuriser 1 \
    --max_processes_classifier 1 \
    --max_batch_memory 1.0 \
    --use_gpu "no"
