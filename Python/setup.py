#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti
#
# This file is part of ASEMI-segmenter.
#
# ASEMI-segmenter is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ASEMI-segmenter is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASEMI-segmenter.  If not, see <http://www.gnu.org/licenses/>.

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
