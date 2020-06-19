#!/bin/bash

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
             --max_processes 1 \
             --max_batch_memory 1 \
             --save_as "$output" \
             --use_gpu "$gpu"
      fi
   done
done
