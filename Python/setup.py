import setuptools

setuptools.setup(
        name='asemi_segmenter',
        version='0.0.0',
        packages=[
                'asemi_segmenter',
                'asemi_segmenter.lib',
                'asemi_segmenter.resources',
                'asemi_segmenter.resources.json_schema'
            ],
        package_data={
                'asemi_segmenter': ['resources/json_schema/*.json']
            },
        install_requires=[
                'fast-histogram',
                'h5py',
                'joblib',
                'loky',
                'numpy',
                'Pillow',
                'scikit-learn',
                'scikit-image',
                'scipy',
                'tqdm',
                'psutil',
                'jsonschema',
                'jsonref',
                'memory_profiler'
            ]
    )