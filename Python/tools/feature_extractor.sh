python feature_extractor.py --volume_fullfname "../example_volume/output/preprocess/volume.hdf" --feature_config_fullfname "feature_config.json" --volume_slice_index 0 --max_processes 1 --max_batch_memory 1 --save_as "nogpu.txt" --use_gpu "no"

python feature_extractor.py --volume_fullfname "../example_volume/output/preprocess/volume.hdf" --feature_config_fullfname "feature_config.json" --volume_slice_index 0 --max_processes 1 --max_batch_memory 1 --save_as "gpu.txt" --use_gpu "yes"
