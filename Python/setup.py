import setuptools
from setuptools_scm import get_version

with open('asemi_segmenter/version.txt', 'w', encoding='utf-8') as f:
    version_string = get_version(root="..", relative_to=__file__,
            local_scheme="node-and-timestamp")
    f.write(version_string)
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.split('==')[0] for line in f.read().strip().split('\n')]

setuptools.setup(
        name='asemi_segmenter',
        version=version_string,
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
