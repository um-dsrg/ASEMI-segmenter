python profiler.py ^
    --volume_fullfname "..\..\example_volume\output\preprocess\volume.hdf" ^
    --segmenter_fullfname "..\..\example_volume\output\train\segmenter.pkl" ^
    --segment_config_fullfname "segment_config.json" ^
    --results_dir "..\..\example_volume\output\segment" ^
    --profiler_results_fullfname "results.txt" ^
    --max_processes_featuriser 1 ^
    --max_processes_classifier 1 ^
    --max_batch_memory 1.0 ^
    --use_gpu "no"
