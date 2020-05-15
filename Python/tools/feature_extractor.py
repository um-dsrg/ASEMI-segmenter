import numpy as np
import memory_profiler
from asemi_segmenter.lib import times
from asemi_segmenter.lib import featurisers
from asemi_segmenter.lib import volumes
from asemi_segmenter.lib import arrayprocs


#########################################
def extract_features(preproc_volume_fullfname, featuriser_config, volume_slice_index, max_processes, max_batch_memory, save_as=None):
    full_volume = volumes.FullVolume(preproc_volume_fullfname)
    full_volume.load()
    
    featuriser = featurisers.load_featuriser_from_config(featuriser_config)
    
    best_block_shape = arrayprocs.get_optimal_block_size(
        full_volume.get_shape(),
        full_volume.get_dtype(),
        featuriser.get_context_needed(),
        max_processes,
        max_batch_memory,
        implicit_depth=True
        )
    
    result = []
    def f(result):
        result.append(
            featuriser.featurise_slice(
                full_volume.get_scale_arrays(featuriser.get_scales_needed()),
                slice_index=volume_slice_index,
                block_rows=best_block_shape[0],
                block_cols=best_block_shape[1],
                n_jobs=max_processes
                )
            )
    
    with times.Timer() as timer:
        mem_usage = memory_profiler.memory_usage((f, (result,)))
    feature_vectors = result[0]
    
    print('duration:', round(timer.duration, 1), 's')
    print('memory:', round(max(mem_usage), 2), 'MB')
    
    if save_as is not None:
        if save_as.endswith('.txt'):
            with open(save_as, 'w', encoding='utf-8') as f:
                for feature_vector in feature_vectors.tolist():
                    print(*feature_vector, sep='\t', file=f)
        if save_as.endswith('.npy'):
            with open(save_as, 'wb') as f:
                np.save(f, feature_vectors, allow_pickle=False)


#########################################
featuriser_config = {
		"type": "composite",
		"params": {
			"featuriser_list": [
				{
					"type": "voxel",
					"params": {}
				},
				{
					"type": "histogram",
					"params": {
						"radius": 19,
						"scale": 4,
						"num_bins": 32
					}
				},
				{
					"type": "lbp",
					"params": {
						"orientation": "front",
						"radius": 25,
						"scale": 4
					}
				},
				{
					"type": "lbp",
					"params": {
						"orientation": "flat",
						"radius": 25,
						"scale": 4
					}
				}
			]
		}
}

print('Running...')

extract_features(
    '../example_volume/preprocess/volume.hdf',
    featuriser_config,
    volume_slice_index=0,
    save_as='features.txt', #Can be a full file name to a text file (.txt) or to a numpy file (.npy) or None
    max_processes=2,
    max_batch_memory=0.1
    )

input('Press enter to exit.')
