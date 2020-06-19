python feature_extractor.py ^
    --volume_fullfname "..\..\example_volume\output\preprocess\volume.hdf" ^
    --feature_config_fullfname "feature_config-histogram.json" ^
    --volume_slice_index 0 ^
    --volume_slice_count 10 ^
    --max_processes 1 ^
    --max_batch_memory 1 ^
    --save_as "results-histogram-gpu=no.txt" ^
    --use_gpu "no"
