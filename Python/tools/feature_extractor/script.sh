#!/bin/bash
#
# Copyright © 2020 Johann A. Briffa
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

for feature in histogram lbp; do
   for gpu in no yes; do
      output="results-$feature-gpu=$gpu.txt"
      if [ ! -f "$output" ]; then
         echo "Extracting features: $feature, gpu=$gpu..."
         python feature_extractor.py \
             --volume_fullfname "../../example_volume/output/preprocess/volume.hdf" \
             --feature_config_fullfname "feature_config-$feature.json" \
             --volume_slice_index 0 \
             --volume_slice_count 10 \
             --max_processes_featuriser 1 \
             --max_batch_memory 1 \
             --save_as "$output" \
             --use_gpu "$gpu"
      fi
   done
done
