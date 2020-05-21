import setuptools

with open('asemi_segmenter/version.txt', 'r', encoding='utf-8') as f:
    version = f.read().strip()
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.split('==')[0] for line in f.read().strip().split('\n')]

setuptools.setup(
        name='asemi_segmenter',
        version=version,
        packages=[
                'asemi_segmenter',
                'asemi_segmenter.lib',
                'asemi_segmenter.resources',
                'asemi_segmenter.resources.cuda',
                'asemi_segmenter.resources.json_schema'
            ],
        package_data={
                'asemi_segmenter': [
                    'version.txt',
                    'resources/json_schema/*.json',
                    'resources/colours/*.json',
                    'resources/cuda/*.cu',
                    ]
            },
        install_requires=requirements
    )
