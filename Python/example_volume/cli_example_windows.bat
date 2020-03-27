python ..\bin\preprocess.py ^
    --volume_dir "volume" ^
    --config_fullfname "preprocess_config.json" ^
    --result_data_fullfname "preprocess\volume.hdf" ^
    --max_processes 2 ^
    --max_batch_memory 0.1

python ..\bin\train.py ^
    --preproc_volume_fullfname "preprocess\volume.hdf" ^
    --subvolume_dir "train\subvolume" ^
    --label_dirs ^
        "train\labels\air" ^
        "train\labels\terracotta" ^
    --config_fullfname "train_config.json" ^
    --result_model_fullfname "train\model.pkl" ^
    --trainingset_file_fullfname "train\trainingset.hdf" ^
    --max_processes 2 ^
    --max_batch_memory 0.1

python ..\bin\evaluate.py ^
    --model_fullfname "train\model.pkl" ^
    --preproc_volume_fullfname "preprocess\volume.hdf" ^
    --subvolume_dir "evaluate\subvolume" ^
    --label_dirs ^
        "evaluate\labels\air" ^
        "evaluate\labels\terracotta" ^
    --results_fullfname "evaluate\results.txt" ^
    --max_processes 2 ^
    --max_batch_memory 0.1

python ..\bin\segment.py ^
    --model_fullfname "train\model.pkl" ^
    --preproc_volume_fullfname "preprocess\volume.hdf" ^
    --config_fullfname "segment_config.json" ^
    --results_dir "segment" ^
    --max_processes 2 ^
    --max_batch_memory 0.1