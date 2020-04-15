import setuptools

setuptools.setup(
        name='asemi_segmenter',
        version=open('asemi_segmenter/version.txt', 'r', encoding='utf-8').read(),
        packages=[
                'asemi_segmenter',
                'asemi_segmenter.lib',
                'asemi_segmenter.resources',
                'asemi_segmenter.resources.json_schema'
            ],
        package_data={
                'asemi_segmenter': [
                    'version.txt',
                    'resources/json_schema/*.json'
                    ]
            },
        install_requires=[
                'fast-histogram==0.7',
                'h5py===2.10.0',
                'joblib==0.14.0',
                'jsonref==0.2',
                'jsonschema==3.2.0',
                'loky==2.6.0',
                'memory_profiler==0.55.0',
                'numpy==1.15.4',
                'Pillow==6.2.1',
                'psutil==5.6.7',
                'scikit-image==0.14.1',
                'scikit-learn==0.19.1',
                'scipy==1.1.0',
                'tqdm==4.43.0'
            ]
    )