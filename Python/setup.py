import setuptools

setuptools.setup(
        name='asemi_segmenter',
        version='0.0.0',
        packages=[
                'asemi_segmenter',
                'asemi_segmenter.lib',
            ],
        install_requires=[
                'fast-histogram',
                'h5py',
                'joblib',
                'loky',
                'numpy',
                'scikit-image',
                'scikit-learn',
                'scipy',
                'tqdm',
                'psutil'
            ]
    )