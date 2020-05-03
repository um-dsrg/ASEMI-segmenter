python ../bin/preprocess.py \
    --volume_dir "volume" \
    --config_fullfname "preprocess_config.json" \
    --result_data_fullfname "preprocess/volume.hdf" \
    --max_processes 2 \
    --max_batch_memory 0.1

python ../bin/analyse_data.py \
    --subvolume_dir "tune/train/subvolume" \
    --label_dirs \
        "tune/train/labels/air" \
        "tune/train/labels/terracotta" \
    --config "analyse_data_config.json" \
    --highlight_radius 3 \
    --results_dir "analyse_data"

python ../bin/tune.py \
    --preproc_volume_fullfname "preprocess/volume.hdf" \
    --train_subvolume_dir "tune/train/subvolume" \
    --train_label_dirs \
        "tune/train/labels/air" \
        "tune/train/labels/terracotta" \
    --eval_subvolume_dir "tune/eval/subvolume" \
    --eval_label_dirs \
        "tune/eval/labels/air" \
        "tune/eval/labels/terracotta" \
    --config_fullfname "tune_config.json" \
    --search_results_fullfname "tune/results.txt" \
    --best_result_fullfname "tune/result.json" \
    --max_processes 2 \
    --max_batch_memory 0.1

python ../bin/train.py \
    --preproc_volume_fullfname "preprocess/volume.hdf" \
    --subvolume_dir "train/subvolume" \
    --label_dirs \
        "train/labels/air" \
        "train/labels/terracotta" \
    --config_fullfname "train_config.json" \
    --result_segmenter_fullfname "train/segmenter.pkl" \
    --trainingset_file_fullfname "train/trainingset.hdf" \
    --max_processes 2 \
    --max_batch_memory 0.1

python ../bin/evaluate.py \
    --segmenter_fullfname "train/segmenter.pkl" \
    --preproc_volume_fullfname "preprocess/volume.hdf" \
    --subvolume_dir "evaluate/subvolume" \
    --label_dirs \
        "evaluate/labels/air" \
        "evaluate/labels/terracotta" \
    --results_fullfname "evaluate/results.txt" \
    --max_processes 2 \
    --max_batch_memory 0.1

python ../bin/segment.py \
    --segmenter_fullfname "train/segmenter.pkl" \
    --preproc_volume_fullfname "preprocess/volume.hdf" \
    --config_fullfname "segment_config.json" \
    --results_dir "segment" \
    --max_processes 2 \
    --max_batch_memory 0.1
