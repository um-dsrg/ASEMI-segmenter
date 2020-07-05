python evaluate_masks.py ^
    --groundtruth_labels ^
        "../../example_volume\volume\all_labels_(not_expected)/air" ^
        "../../example_volume\volume\all_labels_(not_expected)/bones" ^
        "../../example_volume\volume\all_labels_(not_expected)/tissues" ^
    --predicted_labels ^
        "../../example_volume\output\segment\air" ^
        "../../example_volume\output\segment\bones" ^
        "../../example_volume\output\segment\tissues" ^
    --results_dir "results"
