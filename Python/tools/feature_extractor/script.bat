FOR %%a IN (histogram lbp) DO (
   FOR %%b in (no yes) DO (
      SET o=results-%%a-gpu=%%b.txt
      IF NOT EXIST %%o (
         python feature_extractor.py ^
             --volume_fullfname "..\..\example_volume\output\preprocess\volume.hdf" ^
             --feature_config_fullfname "feature_config-%%a.json" ^
             --volume_slice_index 0 ^
             --volume_slice_count 10 ^
             --max_processes_featuriser 1 ^
             --max_batch_memory 1 ^
             --save_as "%%o" ^
             --use_gpu "%%b"
      )
   )
)
